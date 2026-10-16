#!/usr/bin/env python3
"""Unit tests for the LoCoMo parallel QA path (post qi-bug fix, commit 63e39ea).

Validates exactly the code that crashed the 2026-10-15 full run:
- run_one_locomo_conversation: (qi, qa) tuple unpacking, per-question flush,
  resume dedup, build-error records
- run_locomo: a worker raising is consumed by as_completed (no zombie pool)
  and re-raised after ALL futures complete
- concurrent workers writing through the shared write_lock

No LLM calls (FakeSystem). Run with the simplemem venv python.
"""
import json
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, "/home/apulis-dev/code/SimpleMem/eval")
import run_simplemem_eval as R

FAKE_CONV = {
    "speaker_a": "Alice",
    "speaker_b": "Bob",
    "session_1": [
        {"speaker": "A", "text": "I went to the park yesterday."},
        {"speaker": "B", "text": "That sounds nice."},
    ],
    "session_1_date_time": "1:00 pm on 3 May, 2023",
}


def make_qa_rows():
    return [
        (18, {"question": "Q18?", "category": 1, "answer": "A18", "evidence": []}),
        (25, {"question": "Q25?", "category": 2, "answer": "A25", "evidence": []}),
    ]


class FakeMemBuilder:
    failed_windows = 0


class FakeSystem:
    def __init__(self, *a, **k):
        self.memory_builder = FakeMemBuilder()

    def add_dialogue(self, speaker, text, dt):
        pass

    def finalize(self):
        pass

    def ask(self, question):
        time.sleep(0.05)  # force thread interleaving
        return f"answer({question})"


class RaisingBuildSystem(FakeSystem):
    def finalize(self):
        raise RuntimeError("boom-build")


ARGS = types.SimpleNamespace(run_model="fake-model")


def read(p):
    return [json.loads(line) for line in open(p)]


def test_happy_path(tmp):
    pred = tmp / "t1.jsonl"
    done = set()
    ci, n = R.run_one_locomo_conversation(
        0, ARGS, {"conversation": FAKE_CONV}, make_qa_rows(), tmp,
        FakeSystem, done, threading.Lock(), pred)
    assert (ci, n) == (0, 2), f"expected (0,2), got {(ci, n)}"
    rows = read(pred)
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    assert [r["sample_id"] for r in rows] == ["locomo-0-qa-18", "locomo-0-qa-25"]
    assert all(r["prediction"].startswith("answer(") for r in rows)
    assert all(r["error"] == "" for r in rows)
    assert rows[0]["conversation_id"] == 0
    assert done == {"locomo-0-qa-18", "locomo-0-qa-25"}
    print("PASS test1 happy-path (qi unpacking, per-question flush)")


def test_resume_dedup(tmp):
    pred = tmp / "t1.jsonl"  # rows from test1
    done = {"locomo-0-qa-18", "locomo-0-qa-25"}
    ci, n = R.run_one_locomo_conversation(
        0, ARGS, {"conversation": FAKE_CONV}, make_qa_rows(), tmp,
        FakeSystem, done, threading.Lock(), pred)
    assert (ci, n) == (0, 0), f"expected (0,0) on full resume, got {(ci, n)}"
    assert len(read(pred)) == 2
    pred2 = tmp / "t2.jsonl"
    ci, n = R.run_one_locomo_conversation(
        0, ARGS, {"conversation": FAKE_CONV}, make_qa_rows(), tmp,
        FakeSystem, {"locomo-0-qa-18"}, threading.Lock(), pred2)
    assert (ci, n) == (0, 1)
    assert [r["sample_id"] for r in read(pred2)] == ["locomo-0-qa-25"]
    print("PASS test2 resume dedup (full + partial)")


def test_build_error_records(tmp):
    pred = tmp / "t3.jsonl"
    ci, n = R.run_one_locomo_conversation(
        1, ARGS, {"conversation": FAKE_CONV}, make_qa_rows(), tmp,
        RaisingBuildSystem, set(), threading.Lock(), pred)
    assert (ci, n) == (1, 2)
    rows = read(pred)
    assert [r["sample_id"] for r in rows] == ["locomo-1-qa-18", "locomo-1-qa-25"]
    assert all(r["error"] and r["prediction"].startswith("[ERROR]") for r in rows)
    print("PASS test3 build-error records")


def test_concurrent_workers(tmp):
    pred = tmp / "t4.jsonl"
    done = set()
    lock = threading.Lock()
    qa_rows = [(i, {"question": f"Q{i}?", "category": 1, "answer": f"A{i}",
                    "evidence": []}) for i in range(3)]
    results = {}

    def work(ci):
        results[ci] = R.run_one_locomo_conversation(
            ci, ARGS, {"conversation": FAKE_CONV}, qa_rows, tmp,
            FakeSystem, done, lock, pred)

    threads = [threading.Thread(target=work, args=(ci,)) for ci in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(v == (ci, 3) for ci, v in results.items()), results
    rows = read(pred)
    assert len(rows) == 12, f"expected 12 rows, got {len(rows)}"
    sids = [r["sample_id"] for r in rows]
    assert len(set(sids)) == 12, "duplicate sids under concurrency!"
    assert len(done) == 12
    print("PASS test4 concurrent workers (write_lock, no lost/dup rows)")


def test_worker_exception_hardening(tmp):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        time.sleep(0.05)
        raise RuntimeError("worker-boom")

    orig = R.run_one_locomo_conversation
    R.run_one_locomo_conversation = boom
    try:
        locomo_data = [
            {"conversation": FAKE_CONV, "qa": [
                {"question": f"c{ci}-q{i}", "category": 1, "answer": "x",
                 "evidence": []} for i in range(2)]}
            for ci in range(2)
        ]
        args = types.SimpleNamespace(
            run_model="fake", qa_per_category=1, seed=42, resume=False,
            limit_conv=2, locomo_concurrency=2)
        raised = None
        try:
            R.run_locomo(args, locomo_data, tmp, set(), tmp / "t5.jsonl")
        except RuntimeError as e:
            raised = e
        assert raised is not None and "worker-boom" in str(raised), \
            f"expected RuntimeError('worker-boom'), got {raised!r}"
        assert len(calls) == 2, f"expected both futures consumed, got {len(calls)}"
        print("PASS test5 worker exception consumed + re-raised (no zombie pool)")
    finally:
        R.run_one_locomo_conversation = orig


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="locomo_qa_test_"))
    test_happy_path(tmp)
    test_resume_dedup(tmp)
    test_build_error_records(tmp)
    test_concurrent_workers(tmp)
    test_worker_exception_hardening(tmp)
    print(f"ALL TESTS PASSED (tmp={tmp})")


if __name__ == "__main__":
    main()
