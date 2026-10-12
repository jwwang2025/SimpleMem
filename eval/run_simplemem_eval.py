"""
SimpleMem evaluation on LoCoMo (5 QA categories) + LoCoMo-Plus (Cognitive).

Pipeline
--------
1. For every LoCoMo conversation: build ONE SimpleMem memory store from all
   sessions (chronologically, with absolute timestamps), then ask every QA
   question against the same store.
2. For every LoCoMo-Plus sample: stitch cue dialogue + trigger query into a
   LoCoMo conversation on the official time axis (data/build_conv.py), build an
   independent memory store, and respond to the final trigger turn.
3. Score predictions three ways:
   - token F1 (SQuAD/LoCoMo convention, gold answers only)
   - BLEU (sentence-smoothed macro + corpus; gold answers only)
   - LLM-as-a-judge accuracy over ALL six categories, using the official
     Locomo-Plus rubric templates (correct=1 / partial=0.5 / wrong=0).

Outputs (in --out-dir)
----------------------
predictions.jsonl   one record per sample (official judge schema + tracing ids)
judged.jsonl        predictions enriched with judge_label / judge_reason / score
summary.json        overall and per-category metrics

Examples
--------
# tiny smoke run (needs OPENAI_API_KEY in config.py or environment)
python eval/run_simplemem_eval.py --limit-conv 1 --qa-per-category 1 --cognitive-limit 1

# full benchmark, judge with a cheaper model, resumable
python eval/run_simplemem_eval.py --judge-model gpt-4o-mini --judge-concurrency 8 --resume

# predictions + lexical metrics only
python eval/run_simplemem_eval.py --skip-judge

# LoCoMo-Plus (Cognitive) only, 8 samples in parallel, resumable
python eval/run_simplemem_eval.py --skip-locomo --cognitive-concurrency 8 --resume
"""
import argparse
import gc
import importlib.util
import io
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(DATA_DIR))

from locomo_metrics import aggregate_lexical  # noqa: E402
from eval_logging import setup_run_logging  # noqa: E402

log = logging.getLogger("eval")

# Category id -> name. Verified against the data itself and the LoCoMo paper:
# 1 single-hop (282), 2 temporal (321), 3 multi-hop (96),
# 4 open-domain (841), 5 adversarial (446, gold answer almost always empty).
LOCOMO_CATEGORY = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}
COGNITIVE = "Cognitive"

# Official Locomo-Plus judge templates call open-domain "common-sense".
JUDGE_TEMPLATE_KEY = {
    "single-hop": "single-hop",
    "temporal": "temporal",
    "multi-hop": "multi-hop",
    "open-domain": "common-sense",
    "adversarial": "adversarial",
    "Cognitive": "Cognitive",
}

# Judge rubrics adapted from
# https://github.com/xjtuleeyf/Locomo-Plus/blob/main/evaluation_framework/task_eval/prompt.py
JUDGE_TEMPLATES = {
    "multi-hop": """You are a Fact-Checking Judge.
Compare the model's prediction with the reference answer (multi-hop fact QA).

Labels:
- "correct": The answer matches the reference entities (names, places, times) exactly.
- "partial": The answer misses some details or contains minor inaccuracies but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return strictly in JSON:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "single-hop": """You are a Fact-Checking Judge.
Compare the model's prediction with the reference answer (single-hop fact QA).

Labels:
- "correct": The answer matches the reference entities exactly.
- "partial": The answer misses some details but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return strictly in JSON:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "temporal": """You are a Temporal Logic Judge.
Check the calculation, duration, or sequence of events.

Labels:
- "correct": The calculated time, duration, or date matches the reference exactly (semantic equivalents allowed).
- "wrong": The calculation is incorrect, the sequence is reversed, or the specific time is wrong.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return strictly in JSON:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",
    "common-sense": """You are a Knowledge Logic Judge.
Assess whether the prediction applies correct commonsense/world knowledge consistent with the reference.

Labels:
- "correct": The logic and inference are sound and match the reference conclusion.
- "partial": The reasoning is mostly correct but the final conclusion is vague or slightly off.
- "wrong": The reasoning contradicts commonsense or the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return strictly in JSON:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
    "adversarial": """You are a Skeptical Judge evaluating robustness.
The question is inherently misleading (it asks about something not in the conversation).
Judge whether the model's answer conveys that "this was not mentioned in the conversation" (or equivalent refusal).

Labels:
- "correct": The prediction clearly conveys the information was not mentioned / cannot be answered from the conversation.
- "wrong": The prediction gives a concrete answer or otherwise fails to refuse.

Model Prediction:
{pred}

Return strictly in JSON:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",
    "Cognitive": """You are a Memory Awareness Judge.
Judge whether the Model Prediction considers or is linked to the Evidence. A clear connection = correct (1 point); no connection = wrong (0).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint).
- "wrong": The prediction does not show such a link to the evidence.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return strictly in JSON:
{{"label": "correct"|"wrong", "reason": "<does the prediction relate to the evidence?>"}}
""",
}


# ---------------------------------------------------------------------------
# Time parsing (LoCoMo uses e.g. "1:56 pm on 8 May, 2023")
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"], start=1)}
_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([a-z]+),?\s+(\d{4})",
    re.IGNORECASE,
)


def parse_locomo_time(text: str) -> datetime:
    m = _TIME_RE.search(text.strip())
    if not m:
        raise ValueError(f"Unparseable LoCoMo timestamp: {text!r}")
    hour, minute, ampm, day, month, year = m.groups()
    hour = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
    return datetime(int(year), _MONTHS[month.lower()], int(day), hour, int(minute))


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Feeding plans (pure functions; no SimpleMem dependency so they are testable)
# ---------------------------------------------------------------------------

def build_locomo_turns(conv: dict):
    """Yield (speaker, text, datetime) for every turn of one LoCoMo conversation."""
    from build_conv import analyze_conversation

    speaker_a, speaker_b, sessions, session_times = analyze_conversation(conv)
    for turns, dt in zip(sessions, session_times):
        for turn in turns:
            text = (turn.get("text") or "").strip()
            caption = turn.get("blip_caption")
            if caption:
                text = f'{text} [shared an image: {caption}]'.strip()
            if text:
                yield turn.get("speaker", "?"), text, dt


def locomo_evidence_text(conv: dict, raw_evidence) -> str:
    from unified_input import _evidence_to_text, _parse_evidence_list

    return _evidence_to_text(conv, _parse_evidence_list(raw_evidence))


def build_cognitive_plan(plus_item: dict, locomo_item: dict):
    """
    Reproduce data/build_conv.build_context, keeping datetimes.
    Returns (turns_in, question_speaker, question_text, respond_as, evidence_text).
    The final trigger turn is the question; everything before it is memorized.
    """
    from build_conv import (
        analyze_conversation,
        compute_insertion,
        parse_ab_dialogue,
        map_speaker,
    )
    from unified_input import _cue_dialogue_to_evidence

    conv = locomo_item["conversation"]
    speaker_a, speaker_b, sessions, session_times = analyze_conversation(conv)
    _, cue_time, query_time = compute_insertion(session_times, plus_item["time_gap"])

    cue_turns = map_speaker(parse_ab_dialogue(plus_item["cue_dialogue"]), speaker_a, speaker_b)
    query_turns = map_speaker(parse_ab_dialogue(plus_item["trigger_query"]), speaker_a, speaker_b)

    events = [(t, sess) for sess, t in zip(sessions, session_times)]
    events.append((cue_time, cue_turns))
    events.append((query_time, query_turns))
    events.sort(key=lambda x: x[0])

    flat = [(dt, t) for dt, turns in events for t in turns]
    if not flat:
        raise ValueError("Empty stitched cognitive dialogue")
    final_dt, final_turn = flat[-1]

    turns_in = [(t.get("speaker", "?"), (t.get("text") or "").strip(), dt)
                for dt, t in flat[:-1] if (t.get("text") or "").strip()]
    q_speaker = final_turn.get("speaker", "A")
    respond_as = speaker_b if q_speaker == speaker_a else speaker_a
    question_text = (final_turn.get("text") or "").strip()
    evidence = _cue_dialogue_to_evidence(plus_item.get("cue_dialogue", ""), locomo_item)
    return turns_in, q_speaker, question_text, respond_as, evidence


# ---------------------------------------------------------------------------
# SimpleMem system loading
# ---------------------------------------------------------------------------

def load_simplemem_system_cls():
    spec = importlib.util.spec_from_file_location("simplemem_main", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleMemSystem


def build_system(SystemCls, db_path: Path, clear: bool = True):
    db_path.mkdir(parents=True, exist_ok=True)
    return SystemCls(db_path=str(db_path), clear_db=clear)


def release_system(system):
    try:
        backend = getattr(system.vector_store, "backend", None)
        for attr in ("table", "db", "_connection"):
            obj = getattr(backend, attr, None)
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
    except Exception:
        pass
    del system
    gc.collect()


def force_rmtree(path: Path, retries: int = 3):
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except Exception:
            time.sleep(1.0)
    log.warning("could not remove %s (Windows lock?), left on disk", path)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_qa(qa_list, per_category, seed):
    if not per_category:
        return list(enumerate(qa_list))
    rng = random.Random(seed)
    groups = {}
    for idx, qa in enumerate(qa_list):
        groups.setdefault(qa.get("category"), []).append((idx, qa))
    picked = []
    for cat, rows in groups.items():
        rng.shuffle(rows)
        picked.extend(rows[:per_category])
    picked.sort(key=lambda x: x[0])
    return picked


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows):
    with io.open(path, "w" if not path.exists() else "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prediction stages
# ---------------------------------------------------------------------------

def run_locomo(args, locomo_data, out_dir, done_ids, predictions_path):
    SystemCls = load_simplemem_system_cls()
    n_conv = len(locomo_data) if not args.limit_conv else min(args.limit_conv, len(locomo_data))
    for ci in range(n_conv):
        item = locomo_data[ci]
        conv = item["conversation"]
        qa_rows = sample_qa(item.get("qa") or [], args.qa_per_category, args.seed + ci)
        expected_ids = {f"locomo-{ci}-qa-{qi}" for qi, _ in qa_rows}
        if args.resume and expected_ids and expected_ids <= done_ids:
            log.info("[LoCoMo] conv %d: all %d predictions exist, skipping memory build",
                     ci, len(expected_ids))
            continue

        db_path = out_dir / "db" / f"conv_{ci}"
        log.info("[LoCoMo] conv %d: building memory (%s / %s) -> %s",
                 ci, conv.get("speaker_a"), conv.get("speaker_b"), db_path)
        t_build = time.time()
        system = None
        build_error = ""
        try:
            system = build_system(SystemCls, db_path, clear=True)
            gen = build_locomo_turns(conv)
            n_turns = 0
            for speaker, text, dt in gen:
                system.add_dialogue(speaker, text, iso(dt))
                n_turns += 1
            system.finalize()
            failed_windows = getattr(system.memory_builder, "failed_windows", 0)
            if failed_windows:
                raise RuntimeError(
                    f"memory extraction failed for {failed_windows} window(s)")
            log.info("[LoCoMo] conv %d: memory built in %.1fs from %d turns",
                     ci, time.time() - t_build, n_turns)
        except Exception as exc:
            build_error = str(exc)
            log.exception("[LoCoMo] conv %d: memory build failed", ci)

        if build_error:
            # Every QA of this conversation is invalid: record errors so they
            # are excluded from metrics instead of answered from partial memory.
            if system is not None:
                release_system(system)
            err_rows = []
            for qi, qa in qa_rows:
                sid = f"locomo-{ci}-qa-{qi}"
                if sid in done_ids:
                    continue
                err_rows.append({
                    "sample_id": sid,
                    "source": "locomo",
                    "conversation_id": ci,
                    "question_input": qa.get("question", ""),
                    "evidence": locomo_evidence_text(conv, qa.get("evidence") or []),
                    "category": LOCOMO_CATEGORY.get(qa.get("category"), f'category_{qa.get("category")}'),
                    "ground_truth": "" if qa.get("answer") is None else str(qa["answer"]),
                    "prediction": f"[ERROR] {build_error}",
                    "model": args.run_model,
                    "elapsed_sec": 0.0,
                    "error": build_error,
                })
                done_ids.add(sid)
            append_jsonl(predictions_path, err_rows)
            log.error("[LoCoMo] conv %d: wrote %d error records (memory build failed)",
                      ci, len(err_rows))
            continue

        new_rows = []
        t_ask = time.time()
        for qi, qa in qa_rows:
            sid = f"locomo-{ci}-qa-{qi}"
            if sid in done_ids:
                continue
            category = LOCOMO_CATEGORY.get(qa.get("category"), f'category_{qa.get("category")}')
            question = qa.get("question", "")
            t0 = time.time()
            error = ""
            try:
                prediction = system.ask(question)
            except Exception as exc:
                error = str(exc)
                prediction = f"[ERROR] {exc}"
                log.exception("[LoCoMo] ask failed | %s | Q=%s", sid, question[:150])
            record = {
                "sample_id": sid,
                "source": "locomo",
                "conversation_id": ci,
                "question_input": question,
                "evidence": locomo_evidence_text(conv, qa.get("evidence") or []),
                "category": category,
                "ground_truth": "" if qa.get("answer") is None else str(qa["answer"]),
                "prediction": prediction,
                "model": args.run_model,
                "elapsed_sec": round(time.time() - t0, 2),
                "error": error,
            }
            new_rows.append(record)
            done_ids.add(sid)
        append_jsonl(predictions_path, new_rows)
        log.info("[LoCoMo] conv %d: wrote %d new predictions in %.1fs",
                 ci, len(new_rows), time.time() - t_ask)
        release_system(system)


def run_one_cognitive_sample(pi, args, plus_data, locomo_data, out_dir, SystemCls):
    """
    Build an independent SimpleMem memory store for ONE cognitive sample and
    answer its trigger turn. Every call owns its own system / LanceDB directory,
    so it is safe to invoke concurrently from a ThreadPoolExecutor.

    Returns the prediction record, or None when dialogue stitching fails
    (caller counts it as failed). Never raises: build/ask errors are captured
    into the record's "error" field so one bad sample cannot kill the batch.
    """
    sid = f"cognitive-{pi}"
    plus_item = plus_data[pi]
    # Official unified_input.py selects conversations deterministically by modulo.
    locomo_item = locomo_data[pi % len(locomo_data)]
    ci = pi % len(locomo_data)
    t0 = time.time()
    try:
        turns_in, q_speaker, q_text, respond_as, evidence = build_cognitive_plan(plus_item, locomo_item)
    except Exception:
        log.exception("[Cognitive] %s: stitching plan failed", sid)
        return None

    db_path = out_dir / "db" / f"cognitive_{pi}"
    system = build_system(SystemCls, db_path, clear=True)
    error = ""
    prediction = ""
    try:
        for speaker, text, dt in turns_in:
            system.add_dialogue(speaker, text, iso(dt))
        system.finalize()
        # An incomplete memory store makes the answer meaningless: flag the
        # sample as an error (excluded from metrics) instead of silently
        # answering from partial memory.
        failed_windows = getattr(system.memory_builder, "failed_windows", 0)
        if failed_windows:
            raise RuntimeError(
                f"memory extraction failed for {failed_windows} window(s) after "
                f"all retries; sample flagged as error and excluded from metrics")
        question = (
            f'Continue the conversation naturally. {q_speaker} says: "{q_text}" '
            f"Reply as {respond_as} in 1-3 sentences, showing awareness of relevant prior context."
        )
        prediction = system.ask(question)
    except Exception as exc:
        error = str(exc)
        prediction = f"[ERROR] {exc}"
        log.exception("[Cognitive] build/ask failed | %s", sid)
    finally:
        release_system(system)
        if not args.keep_cognitive_db:
            force_rmtree(db_path)

    return {
        "sample_id": sid,
        "source": "cognitive",
        "conversation_id": ci,
        "question_input": plus_item.get("trigger_query", ""),
        "evidence": evidence,
        "category": COGNITIVE,
        "ground_truth": "",
        "prediction": prediction,
        "model": args.run_model,
        "time_gap": plus_item.get("time_gap", ""),
        "elapsed_sec": round(time.time() - t0, 2),
        "error": error,
    }


def run_cognitive(args, plus_data, locomo_data, out_dir, done_ids, predictions_path):
    if args.skip_cognitive:
        log.info("[Cognitive] skipped (--skip-cognitive)")
        return
    limit = len(plus_data) if args.cognitive_limit is None else min(args.cognitive_limit, len(plus_data))
    # Resolve the pending list up front (main thread) so worker threads never
    # mutate the shared resume set / output file.
    pending = [pi for pi in range(limit)
               if not (args.resume and f"cognitive-{pi}" in done_ids)]
    workers = max(1, args.cognitive_concurrency)
    log.info("[Cognitive] evaluating %d samples (%d pending, %d concurrent workers, "
             "independent memory store each)",
             limit, len(pending), workers)
    SystemCls = load_simplemem_system_cls()
    n_written, n_failed = 0, 0
    t_stage = time.time()

    def _consume(record):
        # Single-writer: only the main thread calls this as futures complete,
        # so JSONL appends / done_ids updates need no lock.
        nonlocal n_written, n_failed
        if record is None:
            n_failed += 1
        else:
            append_jsonl(predictions_path, [record])
            done_ids.add(record["sample_id"])
            n_written += 1
            if record.get("error"):
                n_failed += 1
            log.debug("[Cognitive] %s done in %.1fs",
                      record["sample_id"], record["elapsed_sec"])
        processed = n_written + n_failed
        if processed % 10 == 0 or processed == len(pending):
            elapsed = time.time() - t_stage
            rate = processed / elapsed * 60 if elapsed > 0 else 0.0
            log.info("[Cognitive] %d/%d processed (%d written, %d errors, %.0fs elapsed, "
                     "%.2f samples/min)",
                     processed, len(pending), n_written, n_failed, elapsed, rate)

    if workers == 1:
        # Keep the original deterministic serial path for --cognitive-concurrency 1.
        for pi in pending:
            _consume(run_one_cognitive_sample(
                pi, args, plus_data, locomo_data, out_dir, SystemCls))
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = [pool.submit(run_one_cognitive_sample, pi, args,
                               plus_data, locomo_data, out_dir, SystemCls)
                   for pi in pending]
        try:
            for fut in as_completed(futures):
                _consume(fut.result())
        except KeyboardInterrupt:
            # Persisted records stay valid; --resume skips them on restart.
            log.warning("[Cognitive] interrupted after %d persisted samples; "
                        "cancelling queued work (in-flight requests are left to finish).",
                        n_written)
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    log.info("[Cognitive] stage finished: %d written, %d failed, %.1fs total",
             n_written, n_failed, time.time() - t_stage)


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------

def parse_judge_response(raw: str):
    label, reason = "", ""
    raw = (raw or "").strip()
    m = re.search(
        r'"label"\s*:\s*"([^"]+)"\s*,\s*"reason"\s*:\s*"((?:[^"\\]|\\.)*)"',
        raw,
    )
    if m:
        label, reason = m.group(1).strip(), m.group(2).strip()
    else:
        try:
            obj = json.loads(raw)
            label = str(obj.get("label", "")).strip()
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            low = raw.lower()
            if "correct" in low:
                label = "correct"
            elif "partial" in low:
                label = "partial"
            elif "wrong" in low:
                label = "wrong"
            reason = raw[:200]
    return label, reason


def run_judge(args, predictions, judged_path, done_judged):
    from openai import OpenAI
    from simplemem.core.settings import settings
    from simplemem.core.utils.llm_client import compute_backoff, is_rate_limit_error

    # Precedence: CLI --judge-model > config JUDGE_* > config OPENAI_*/LLM_MODEL
    api_key = settings.JUDGE_API_KEY or settings.OPENAI_API_KEY
    base_url = settings.JUDGE_BASE_URL or settings.OPENAI_BASE_URL
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured (root config.py or environment variable).")
    # max_retries=0: the SDK's short fixed backoff multiplied request volume
    # during 429 storms; judge_one below owns the retry policy (jittered
    # exponential backoff, config-driven budget).
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    model = args.judge_model or settings.JUDGE_MODEL or settings.LLM_MODEL
    # Reasoning models (e.g. DeepSeek-R1) spend completion tokens on the chain
    # of thought; 512 would often leave the final content empty.
    max_tokens = 2048 if "r1" in model.lower() else 512

    pending = [r for r in predictions if r["sample_id"] not in done_judged]
    # Records flagged with prediction-stage errors (LLM outage, incomplete
    # memory build, ...) are NOT judged: they are excluded from all metrics
    # downstream instead of being silently scored as wrong answers.
    n_err_pred = sum(1 for r in predictions if r.get("error"))
    if n_err_pred:
        log.info("[Judge] skipping %d error-flagged predictions (excluded from metrics)",
                 n_err_pred)
    pending = [r for r in predictions
               if not r.get("error") and r["sample_id"] not in done_judged]
    log.info("[Judge] %d samples to judge with %s (concurrency=%d)",
             len(pending), model, args.judge_concurrency)
    if not pending:
        return read_jsonl(judged_path)
    judge_log = logging.getLogger("eval.judge")

    judge_max_retries = max(1, int(getattr(settings, "JUDGE_MAX_RETRIES", 10)))
    judge_base = float(getattr(settings, "JUDGE_RETRY_BASE_DELAY", 2.0))
    judge_cap = float(getattr(settings, "LLM_RETRY_MAX_DELAY", 60.0))
    judge_jitter = float(getattr(settings, "LLM_RETRY_JITTER", 1.0))

    def judge_one(record):
        cat = record.get("category") or "unknown"
        template_key = JUDGE_TEMPLATE_KEY.get(cat)
        template = JUDGE_TEMPLATES.get(template_key)
        if template is None:
            return record, "", "no template", 0.0
        prompt = template.format(
            gold=str(record.get("ground_truth") or ""),
            pred=str(record.get("prediction") or ""),
            evidence=str(record.get("evidence") or ""),
        )
        raw = ""
        for attempt in range(judge_max_retries):
            try:
                # Qwen3 on SiliconFlow/dashscope enables thinking by default; with a
                # small max_tokens the reasoning tokens exhaust the budget and leave
                # content empty -> every sample would be judged wrong. Disable it.
                judge_extra_body = None
                if base_url and (
                    "siliconflow.cn" in base_url or "dashscope.aliyuncs.com" in base_url
                ):
                    judge_extra_body = {"enable_thinking": False}
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body=judge_extra_body,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if not raw:
                    raise RuntimeError(
                        "judge returned empty content (reasoning token budget exhausted?)")
                break
            except Exception as exc:
                if attempt < judge_max_retries - 1:
                    wait = compute_backoff(attempt, judge_base, judge_cap, judge_jitter,
                                           rate_limit=is_rate_limit_error(exc))
                    judge_log.warning(
                        "judge api attempt %d/%d failed | %s | %s; retrying in %.1fs",
                        attempt + 1, judge_max_retries, record.get("sample_id"), exc, wait)
                    time.sleep(wait)
                else:
                    # A judge API failure is NOT evidence that the answer is
                    # wrong. Mark the record with judge_error so it is excluded
                    # from metrics - fabricating "wrong" here is exactly how a
                    # throttled run silently destroys its own accuracy.
                    judge_log.error("judge api failed after %d attempts | %s | %s",
                                    judge_max_retries, record.get("sample_id"), exc)
                    out = dict(record)
                    out["judge_error"] = f"judge api error: {exc}"
                    return out, "error", f"judge api error: {exc}", 0.0
        label, reason = parse_judge_response(raw)
        score = {"correct": 1.0, "partial": 0.5}.get(label.lower(), 0.0)
        out = dict(record)
        out.update(judge_label=label, judge_reason=reason, judge_score=score,
                   judge_model=model)
        return out, label, reason, score

    done = 0
    t0 = time.time()
    with io.open(judged_path, "a", encoding="utf-8") as sink:
        if args.judge_concurrency <= 1:
            for r in pending:
                out, *_ = judge_one(r)
                sink.write(json.dumps(out, ensure_ascii=False) + "\n")
                sink.flush()
                done += 1
                if done % 25 == 0:
                    judge_log.info("progress %d/%d", done, len(pending))
        else:
            with ThreadPoolExecutor(max_workers=args.judge_concurrency) as pool:
                futures = [pool.submit(judge_one, r) for r in pending]
                for fut in as_completed(futures):
                    out, *_ = fut.result()
                    sink.write(json.dumps(out, ensure_ascii=False) + "\n")
                    sink.flush()
                    done += 1
                    if done % 25 == 0:
                        judge_log.info("progress %d/%d (%.0fs)", done, len(pending), time.time() - t0)
    log.info("[Judge] wrote %d new judged records in %.1fs", done, time.time() - t0)
    return read_jsonl(judged_path)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(judged):
    # Records flagged with errors (prediction-stage: LLM outage / incomplete
    # memory build; or judge-stage: judge API failure) are EXCLUDED from every
    # metric - an infrastructure failure is not a model mistake. They are still
    # counted and reported via n_error / error_rate.
    valid = [r for r in judged if not r.get("error") and not r.get("judge_error")]
    lexical = aggregate_lexical(valid)
    by_cat = {}
    cats = sorted({r.get("category") for r in judged})
    for cat in cats:
        rows = [r for r in judged if r.get("category") == cat]
        scored = [r for r in rows
                  if "judge_score" in r and not r.get("error") and not r.get("judge_error")]
        lex = lexical["by_category"].get(cat, {"n": 0, "f1": None, "bleu_sentence": None, "bleu_corpus": None, "bleu_1": None, "bleu_corpus_1": None})
        by_cat[cat] = {
            "n_predictions": len(rows),
            "n_judged": len(scored),
            "n_error": sum(1 for r in rows if r.get("error") or r.get("judge_error")),
            "judge_accuracy_strict": round(
                sum(1 for r in scored if r.get("judge_label") == "correct") / len(scored), 4
            ) if scored else None,
            "judge_score_avg": round(
                sum(float(r.get("judge_score", 0)) for r in scored) / len(scored), 4
            ) if scored else None,
            "n_gold": lex["n"],
            "f1": lex["f1"],
            "bleu_sentence": lex["bleu_sentence"],
            "bleu_corpus": lex["bleu_corpus"],
            "bleu_1": lex.get("bleu_1"),
            "bleu_corpus_1": lex.get("bleu_corpus_1"),
        }

    scored_all = [r for r in judged
                  if "judge_score" in r and not r.get("error") and not r.get("judge_error")]
    n_error_all = sum(1 for r in judged if r.get("error") or r.get("judge_error"))
    summary = {
        "overall": {
            "n_predictions": len(judged),
            "n_judged": len(scored_all),
            "n_error": n_error_all,
            "error_rate": round(n_error_all / len(judged), 4) if judged else None,
            "judge_accuracy_strict": round(
                sum(1 for r in scored_all if r.get("judge_label") == "correct") / len(scored_all), 4
            ) if scored_all else None,
            "judge_score_avg": round(
                sum(float(r.get("judge_score", 0)) for r in scored_all) / len(scored_all), 4
            ) if scored_all else None,
            **lexical["overall"],
        },
        "by_category": by_cat,
    }
    return summary


def print_summary(summary):
    o = summary["overall"]
    print("\n" + "=" * 78)
    print("OVERALL  (F1/BLEU on gold-answer samples only; judge over all samples)")
    print("=" * 78)
    print(f"  predictions={o['n_predictions']}  judged={o['n_judged']}  gold={o.get('n', 0)}"
          f"  errors(excluded)={o.get('n_error', 0)}")
    print(f"  judge accuracy (strict)     : {o['judge_accuracy_strict']}")
    print(f"  judge score (partial=0.5)   : {o['judge_score_avg']}")
    print(f"  F1                          : {o.get('f1')}")
    print(f"  BLEU (sentence, smoothed)   : {o.get('bleu_sentence')}")
    print(f"  BLEU (corpus)               : {o.get('bleu_corpus')}")
    print(f"  BLEU-1 (sentence, paper)    : {o.get('bleu_1')}")
    print(f"  BLEU-1 (corpus)             : {o.get('bleu_corpus_1')}")
    print("-" * 78)
    header = f"  {'category':<12}{'n':>5}{'judg':>6}{'judAcc':>9}{'judAvg':>8}{'nGold':>7}{'F1':>8}{'BLEU1':>8}{'err':>5}"
    print(header)
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<12}{s['n_predictions']:>5}{s['n_judged']:>6}"
              f"{str(s['judge_accuracy_strict']):>9}{str(s['judge_score_avg']):>8}"
              f"{s['n_gold']:>7}{str(s['f1']):>8}{str(s['bleu_1']):>8}{s['n_error']:>5}")
    print("=" * 78)
    if o.get("n_error"):
        print("!" * 78)
        print(f"  WARNING: {o['n_error']} record(s) had errors and were EXCLUDED from")
        print( "  every metric above. Fix the cause (usually LLM API rate limiting)")
        print( "  and re-run before trusting these numbers.")
        print("!" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimpleMem evaluation on LoCoMo + LoCoMo-Plus")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-dir", default=str(ROOT / "eval_output"))
    parser.add_argument("--limit-conv", type=int, default=0,
                        help="Use only the first N LoCoMo conversations (0 = all 10)")
    parser.add_argument("--qa-per-category", type=int, default=0,
                        help="Per conversation, sample at most N QA per category (0 = all)")
    parser.add_argument("--cognitive-limit", type=int, default=None,
                        help="Evaluate only the first N LoCoMo-Plus samples (default all 401)")
    parser.add_argument("--cognitive-concurrency", type=int, default=4,
                        help="Number of LoCoMo-Plus samples processed concurrently; each "
                             "builds an independent memory store. Lower it (e.g. 1-2) when "
                             "the LLM/embedding API rate-limits; 1 restores serial mode")
    parser.add_argument("--skip-cognitive", action="store_true")
    parser.add_argument("--skip-locomo", action="store_true",
                        help="Skip original LoCoMo 5-category QA (run Cognitive only)")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-model", default="", help="Default: same LLM_MODEL as the system")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--run-model", default="", help="Label written into records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already present in predictions.jsonl / judged.jsonl")
    parser.add_argument("--keep-cognitive-db", action="store_true",
                        help="Keep per-sample cognitive LanceDB stores (deleted by default)")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 if any record has errors (they are excluded "
                             "from metrics). Use for unattended runs so a throttled or "
                             "half-failed run can never be mistaken for a result.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--run-tag", default="", help="Optional suffix for the log file name")
    args = parser.parse_args()

    if not args.limit_conv:
        args.limit_conv = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger, log_paths = setup_run_logging(out_dir, level=args.log_level, run_tag=args.run_tag or None)
    started_at = datetime.now()

    def _git_sha():
        try:
            out = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() or None
        except Exception:
            return None

    if not args.run_model:
        from simplemem.core.settings import settings
        args.run_model = settings.LLM_MODEL

    # Persist reproducibility metadata next to the logs.
    run_config = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "args": vars(args),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_sha(),
        "logs": log_paths,
    }
    with io.open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    logger.info("Run config saved to %s", out_dir / "run_config.json")
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False))
    if run_config["git_commit"]:
        logger.info("Git commit: %s", run_config["git_commit"])

    data_dir = Path(args.data_dir)
    predictions_path = out_dir / "predictions.jsonl"
    judged_path = out_dir / "judged.jsonl"
    summary_path = out_dir / "summary.json"

    try:
        with io.open(data_dir / "locomo10.json", "r", encoding="utf-8") as f:
            locomo_data = json.load(f)
        with io.open(data_dir / "locomo_plus.json", "r", encoding="utf-8") as f:
            plus_data = json.load(f)
        logger.info("Loaded %d LoCoMo conversations, %d cognitive samples",
                    len(locomo_data), len(plus_data))

        existing = read_jsonl(predictions_path)
        done_ids = {r["sample_id"] for r in existing}
        if args.resume and existing:
            # Error-flagged predictions are retried on resume instead of being
            # kept as permanent holes in the metrics.
            retry_ids = {r["sample_id"] for r in existing if r.get("error")}
            if retry_ids:
                kept = [r for r in existing if not r.get("error")]
                with io.open(predictions_path, "w", encoding="utf-8") as f:
                    for r in kept:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done_ids = {r["sample_id"] for r in kept}
                logger.info("Resuming: dropped %d error-flagged predictions for retry",
                            len(retry_ids))
                judged_path_retry = out_dir / "judged.jsonl"
                if judged_path_retry.exists():
                    jrows = read_jsonl(judged_path_retry)
                    jkept = [r for r in jrows
                             if r["sample_id"] not in retry_ids and not r.get("judge_error")]
                    if len(jkept) != len(jrows):
                        with io.open(judged_path_retry, "w", encoding="utf-8") as f:
                            for r in jkept:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        logger.info("Resuming: dropped %d stale judged rows for retry",
                                    len(jrows) - len(jkept))
        if args.resume:
            logger.info("Resuming: %d predictions already present", len(done_ids))

        if not args.skip_locomo:
            run_locomo(args, locomo_data, out_dir, done_ids, predictions_path)
        run_cognitive(args, plus_data, locomo_data, out_dir, done_ids, predictions_path)

        predictions = read_jsonl(predictions_path)
        n_errors = sum(1 for r in predictions if r.get("error"))
        logger.info("Total predictions: %d (%d flagged with errors)", len(predictions), n_errors)

        judged = existing
        if not args.skip_judge:
            judged = read_jsonl(judged_path)
            done_judged = {r["sample_id"] for r in judged}
            judged = run_judge(args, predictions, judged_path, done_judged)
        else:
            logger.info("LLM-as-a-judge skipped (--skip-judge)")

        # lexical metrics can always be computed from the latest predictions even
        # before judging; after judging we merge any judge fields onto the same rows.
        lex_rows = {r["sample_id"]: r for r in predictions}
        for r in judged:
            if r["sample_id"] in lex_rows and "judge_score" in r:
                lex_rows[r["sample_id"]].update(
                    judge_label=r.get("judge_label"),
                    judge_reason=r.get("judge_reason"),
                    judge_score=r.get("judge_score"),
                    judge_model=r.get("judge_model"),
                )
        merged = list(lex_rows.values())
        summary = summarize(merged)
        with io.open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print_summary(summary)
        n_err = sum(1 for r in merged if r.get("error") or r.get("judge_error"))
        if n_err:
            logger.error("%d record(s) flagged with errors were EXCLUDED from all metrics",
                         n_err)
            if args.strict:
                logger.error("--strict: exiting with code 1; fix the errors and re-run")
                sys.exit(1)
        logger.info("Run finished in %.1fs", (datetime.now() - started_at).total_seconds())
        logger.info("Artifacts:")
        logger.info("  predictions : %s", predictions_path)
        logger.info("  judged      : %s", judged_path if not args.skip_judge else "(skipped)")
        logger.info("  summary     : %s", summary_path)
        logger.info("  run config  : %s", out_dir / "run_config.json")
        logger.info("  logs        : %s", log_paths["run_log"])
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). Partial results and logs were kept.")
        raise
    except Exception:
        logger.exception("Evaluation run failed with an unhandled exception")
        raise


if __name__ == "__main__":
    main()
