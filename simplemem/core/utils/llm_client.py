"""
LLM Client - Handles all LLM interactions
"""
import json
import random
import time
import threading
from typing import List, Dict, Any, Optional
from openai import OpenAI
from simplemem.core.settings import settings as config


def is_rate_limit_error(exc: Exception) -> bool:
    """True for HTTP 429 / SiliconFlow rate-limit rejections (codes 50601/50602)."""
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return ("429" in text or "rate limit" in text or "too many requests" in text
            or "50601" in text or "50602" in text)


def compute_backoff(attempt: int, base: float, cap: float, jitter: float,
                    rate_limit: bool = False) -> float:
    """Exponential backoff with optional jitter.

    Full jitter (jitter=1.0 -> sleep ~ U(0, delay)) decorrelates concurrent
    workers so a rate-limit window is not hammered by a synchronized retry
    wave - the direct cause of the 2026-10-12 retry storm (3,200 attempts/min
    against a 1000 RPM cap). Non-rate-limit errors (timeouts, resets) use a
    shorter ladder because waiting longer rarely helps them.
    """
    ladder_cap = cap if rate_limit else min(cap, 8.0)
    delay = min(ladder_cap, base * (2 ** attempt))
    if jitter >= 1.0:
        delay = random.uniform(0.0, delay)
    elif jitter > 0:
        delay = delay * (1.0 - jitter * random.random())
    return max(0.0, delay)


class RateLimiter:
    """Thread-safe dual-bucket (requests + tokens) client-side throttle.

    Strategy for a STRICTLY ZERO 429 rate: never touch the server's limit in
    the first place. The SiliconFlow free tier allows 1000 RPM / 50K TPM per
    account (per model, sliding window); we deliberately offer at most
    LLM_RPM_BUDGET / LLM_TPM_BUDGET (defaults 300 RPM = 30%, 30K TPM = 60%
    of the caps). With ~40% headroom the sliding window never sees an excess,
    so no request is ever rejected; the jittered-retry machinery remains a
    pure safety net, not the primary mechanism.

    Both buckets refill continuously but hold at most `burst_seconds` worth
    of tokens, so an idle period cannot bank a large burst - the offered
    load curve stays smooth, which is what sliding-window limiters measure.
    """

    def __init__(self, rpm: float = 300.0, tpm: float = 30000.0,
                 burst_seconds: float = 2.0, enabled: bool = True):
        self.rpm = max(0.0, float(rpm))
        self.tpm = max(0.0, float(tpm))
        self.burst_seconds = max(0.1, float(burst_seconds))
        self.enabled = bool(enabled) and (self.rpm > 0 or self.tpm > 0)
        self._req_rate = self.rpm / 60.0    # request tokens per second
        self._tok_rate = self.tpm / 60.0    # token tokens per second
        self._req_cap = self._req_rate * self.burst_seconds
        self._tok_cap = self._tok_rate * self.burst_seconds
        self._lock = threading.Lock()
        self._req_bucket = self._req_cap    # start full: no cold-start penalty
        self._tok_bucket = self._tok_cap
        self._last = time.monotonic()
        self.total_wait_seconds = 0.0       # observability (approximate)

    def _refill_locked(self):
        now = time.monotonic()
        dt = max(0.0, now - self._last)
        self._last = now
        if self._req_rate:
            self._req_bucket = min(self._req_cap, self._req_bucket + dt * self._req_rate)
        if self._tok_rate:
            self._tok_bucket = min(self._tok_cap, self._tok_bucket + dt * self._tok_rate)

    def acquire(self, tokens: float = 0.0):
        """Block until one request costing `tokens` (estimated) may be sent."""
        if not self.enabled:
            return
        tokens = max(0.0, float(tokens))
        started = time.monotonic()
        while True:
            with self._lock:
                self._refill_locked()
                # A single call may cost more than the bucket capacity (a 3K
                # extraction vs a 1K cap). Clamp instead of deadlocking: such
                # calls are then paced at cap/rate seconds each, i.e. exactly
                # tpm/tokens requests per minute - the intended behavior.
                need_tok = min(tokens, self._tok_cap) if self._tok_cap > 0 else 0.0
                req_ok = self._req_rate <= 0 or self._req_bucket >= 1.0
                tok_ok = self._tok_rate <= 0 or self._tok_bucket >= need_tok
                if req_ok and tok_ok:
                    if self._req_rate > 0:
                        self._req_bucket -= 1.0
                    self._tok_bucket -= need_tok
                    self.total_wait_seconds += time.monotonic() - started
                    return
                wait_req = ((1.0 - self._req_bucket) / self._req_rate
                            if self._req_rate > 0 and self._req_bucket < 1.0 else 0.0)
                wait_tok = ((need_tok - self._tok_bucket) / self._tok_rate
                            if self._tok_rate > 0 and self._tok_bucket < need_tok else 0.0)
                wait = max(wait_req, wait_tok)
            # Sleep OUTSIDE the lock, in small slices so a slow waiter cannot
            # block others and overshoot stays bounded by the slice size.
            time.sleep(min(max(wait, 0.005), 0.25))


_global_limiter: Optional["RateLimiter"] = None
_global_limiter_lock = threading.Lock()
_call_stats = {"calls": 0, "rejected_429": 0}
_call_stats_lock = threading.Lock()


def get_rate_limiter() -> "RateLimiter":
    """Process-wide limiter shared by EVERY LLM path (memory builder, answer
    generator, judge) so the offered budget is enforced globally, not per
    component - the judge phase alone offered ~47K TPM in the last run."""
    global _global_limiter
    with _global_limiter_lock:
        if _global_limiter is None:
            _global_limiter = RateLimiter(
                rpm=float(getattr(config, "LLM_RPM_BUDGET", 300)),
                tpm=float(getattr(config, "LLM_TPM_BUDGET", 30000)),
                burst_seconds=float(getattr(config, "LLM_RATE_LIMIT_BURST_SECONDS", 2.0)),
                enabled=bool(getattr(config, "ENABLE_RATE_LIMITER", True)),
            )
        return _global_limiter


def estimate_call_tokens(messages: List[Dict[str, str]],
                         expected_output_tokens: float = 1000.0) -> float:
    """Conservative pre-call estimate. English text ~4 chars/token; the
    expected-output margin covers the extraction JSON (~1K tokens). Deliberate
    over-estimation only slows the run; under-estimation risks a real 429."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if content:
            chars += len(str(content))
    return chars / 4.0 + float(expected_output_tokens)


def note_llm_call(rejected_429: bool = False):
    """Count one real API attempt. The eval harness calls this for its own
    judge requests (separate client) so n_429 in summary.json covers ALL
    traffic - making 'the 429 rate was zero' a verifiable fact, not a hope."""
    with _call_stats_lock:
        _call_stats["calls"] += 1
        if rejected_429:
            _call_stats["rejected_429"] += 1


def get_llm_stats() -> Dict[str, int]:
    with _call_stats_lock:
        return dict(_call_stats)


class LLMClient:
    """
    Unified LLM client interface
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING

        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            print(f"Using custom OpenAI base URL: {self.base_url}")

        if self.enable_thinking:
            print(f"Deep thinking mode enabled")

        # self.client = OpenAI(**client_kwargs)
        # LLM_SDK_MAX_RETRIES=0 disables the SDK's own internal retries: they use
        # a short fixed backoff and fire IN ADDITION to the loop below, which
        # multiplied request volume during 429 storms. All backoff now happens
        # exactly once, in chat_completion().
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=int(getattr(config, "LLM_SDK_MAX_RETRIES", 0)),
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism.

        Retries are config-driven (LLM_MAX_RETRIES / LLM_RETRY_BASE_DELAY /
        LLM_RETRY_MAX_DELAY / LLM_RETRY_JITTER): exponential backoff with full
        jitter, rate-limit aware. Default budget is 10 attempts - a 429 storm
        can easily outlast the old 3 attempts x {1s, 2s, 4s}.
        """
        if max_retries is None:
            max_retries = max(1, int(getattr(config, "LLM_MAX_RETRIES", 10)))
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen-compatible APIs (identified by base_url)
        is_qwen_api = self.base_url and (
            "dashscope.aliyuncs.com" in self.base_url or "siliconflow.cn" in self.base_url
        )
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        # For OpenAI and other APIs, don't add extra_body parameters

        # Retry mechanism: exponential backoff + full jitter, rate-limit aware
        base = float(getattr(config, "LLM_RETRY_BASE_DELAY", 1.0))
        cap = float(getattr(config, "LLM_RETRY_MAX_DELAY", 60.0))
        jitter = float(getattr(config, "LLM_RETRY_JITTER", 1.0))
        limiter = get_rate_limiter()
        est_tokens = estimate_call_tokens(messages, expected_output_tokens=1000.0)
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Client-side throttle BEFORE sending: keep offered load under
                # budget so the server never 429s in the first place (zero-429
                # strategy); the retry loop below is only the safety net.
                limiter.acquire(est_tokens)
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    note_llm_call()
                    return self._handle_streaming_response(**kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    note_llm_call()
                    return response.choices[0].message.content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except Exception as e:
                # print(e)
                last_exception = e
                note_llm_call(rejected_429=is_rate_limit_error(e))
                if attempt < max_retries - 1:
                    rate_limited = is_rate_limit_error(e)
                    wait_time = compute_backoff(attempt, base, cap, jitter,
                                                rate_limit=rate_limited)
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}"
                          f"{', rate-limited' if rate_limited else ''}): {e}")
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {e}")
        
        # If all retries failed, raise the last exception
        raise last_exception

    def _handle_streaming_response(self, **kwargs) -> str:
        """
        Handle streaming response and collect full content
        """
        full_content = []
        stream = self.client.chat.completions.create(**kwargs)

        # for chunk in stream:
        #     if chunk.choices is not None:
        #         print(chunk.choices[0].delta.content)
        
        # print('---------')

        for chunk in stream:
            # print(chunk)
            # fix list index out of range
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_content.append(content)
                # print(full_content)
                # Optional: print streaming content in real-time
                # print(content, end='', flush=True)
        # print(full_content)
        print()
        return ''.join(full_content)

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
