"""
SimpleMem runtime settings.

Resolution order for each attribute:
  1. User's top-level ``config.py`` (if importable on sys.path)
  2. Environment variable of the same name
  3. Built-in default
"""
import os


_DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": None,
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": None,
    "LLM_MODEL": "gpt-4.1-mini",
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "EMBEDDING_DIMENSION": 1024,
    "EMBEDDING_CONTEXT_LENGTH": 32768,
    "ENABLE_THINKING": False,
    "USE_STREAMING": True,
    "USE_JSON_FORMAT": False,
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 4096,
    "WINDOW_SIZE": 40,
    "OVERLAP_SIZE": 2,
    "SEMANTIC_TOP_K": 25,
    "KEYWORD_TOP_K": 5,
    "STRUCTURED_TOP_K": 5,
    "LANCEDB_PATH": "./lancedb_data",
    "MEMORY_TABLE_NAME": "memory_entries",
    "ENABLE_PARALLEL_PROCESSING": True,
    "MAX_PARALLEL_WORKERS": 16,
    "ENABLE_PARALLEL_RETRIEVAL": True,
    "MAX_RETRIEVAL_WORKERS": 8,
    "ENABLE_PLANNING": True,
    "ENABLE_REFLECTION": True,
    "MAX_REFLECTION_ROUNDS": 2,
    # --- LLM retry / rate-limit resilience ---
    # SiliconFlow free tier (Qwen/Qwen3-8B): 1000 RPM / 50K TPM, defined at the
    # ACCOUNT level and per model. A 48-concurrency run without jittered backoff
    # produced a retry storm (~3,200 attempts/min > 1000 RPM) and 99% of chat
    # requests were rejected with 429, silently destroying 93% of extraction
    # windows. These defaults make every LLM call survive rate limiting.
    "LLM_MAX_RETRIES": 10,            # attempts per logical LLM call (was 3)
    "LLM_RETRY_BASE_DELAY": 1.0,      # seconds; exponential base
    "LLM_RETRY_MAX_DELAY": 60.0,      # per-sleep cap for rate-limit errors
    "LLM_RETRY_JITTER": 1.0,          # 1.0 = full jitter sleep ~ U(0, delay)
    "LLM_SDK_MAX_RETRIES": 0,         # disable OpenAI SDK internal retries; we own retries
    "LLM_EXTRACT_MAX_RETRIES": 10,    # memory-builder per-window retries (was 3)
    "LLM_ANSWER_MAX_RETRIES": 10,     # answer-generator retries (was 3)
    "JUDGE_MAX_RETRIES": 10,          # LLM-as-judge retries (was 3)
    "JUDGE_RETRY_BASE_DELAY": 2.0,    # judge backoff base (seconds)
    # --- Client-side rate limiting (zero-429 strategy) ---
    # Retries alone cannot guarantee clean results: during the 2026-10-12 run
    # 99.1% of chat requests were rejected and 93% of extraction windows died
    # anyway. The robust fix is to never EXCEED the limit in the first place:
    # offer at most these budgets - 24K TPM = 48% of the 50K cap (measured safe:
    # the judge phase ran ~47K TPM with zero 429) - leaving headroom for
    # bursts, in-flight calls and the judge phase sharing the same account.
    # NOTE: the static token estimate under-counted Qwen3's real output ~1.5x
    # in the first v3 hours (true offered TPM hit ~70K despite the 30K nominal
    # budget -> 252 x 429). chat_completion now calibrates estimates against
    # real response usage (EWMA), and the budget was lowered to 24K so even a
    # 1.5x estimation error stays under the cap. Cost: the full run needs
    # ~20.2M tokens -> wall clock ~12-16h. Correctness first.
    "ENABLE_RATE_LIMITER": True,      # master switch (False = pass-through)
    "LLM_RPM_BUDGET": 300,            # offered requests/min (server cap 1000)
    "LLM_TPM_BUDGET": 24000,          # offered tokens/min (server cap 50000)
    "LLM_RATE_LIMIT_BURST_SECONDS": 2.0,  # bucket capacity window; smaller = smoother
}


def _coerce(default, raw):
    if default is None or isinstance(default, str):
        return raw
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


class Settings:
    def __init__(self):
        try:
            import config as _user_config
        except ImportError:
            _user_config = None
        self._user_config = _user_config

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so no recursion via _user_config.
        user_cfg = self.__dict__.get("_user_config")
        if user_cfg is not None and hasattr(user_cfg, name):
            return getattr(user_cfg, name)
        env_val = os.getenv(name)
        if name in _DEFAULTS:
            if env_val is not None:
                return _coerce(_DEFAULTS[name], env_val)
            return _DEFAULTS[name]
        if env_val is not None:
            return env_val
        raise AttributeError(
            f"Setting {name!r} is not defined. Set it in config.py or as env var."
        )


settings = Settings()
