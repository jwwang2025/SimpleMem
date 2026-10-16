#!/usr/bin/env python3
"""Unit tests for the LoCoMo parallel QA path (post qi-bug fix, commit 63e39ea)
plus the paper-protocol adversarial (category 5) path.

Validates exactly the code that crashed the 2026-10-15 full run:
- run_one_locomo_conversation: (qi, qa) tuple unpacking, per-question flush,
  resume dedup, build-error records
- run_locomo: a worker raising is consumed by as_completed (no zombie pool)
  and re-raised after ALL futures complete
- concurrent workers writing through the shared write_lock

And the --adversarial-mode paper path (mirrors the SimpleMem paper's
test_locomo10.py):
- category-5 rows: fixed refusal reference, protocol tag, reflection-free
  retrieval (enable_reflection=False), temperature-0.5 binary-choice
  generation, 3-attempt fallback to the refusal string
- open mode keeps the legacy system.ask + empty-gold behaviour
- judge rubric routing (adversarial-paper vs adversarial)

No LLM calls (FakeSystem family). Run with the simplemem venv python.
"""
import json
import sys
import threading
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "eval"))
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
ADV_ARGS_PAPER = types.SimpleNamespace(run_model="fake-model", adversarial_mode="paper")
ADV_ARGS_OPEN = types.SimpleNamespace(run_model="fake-model", adversarial_mode="open")

ADV_QA = {"question": "What did Caroline realize after her charity race?",
          "category": 5, "answer": "",
          "adversarial_answer": "self-care is important", "evidence": []}


class FakeEntry:
    """Minimal MemoryEntry stand-in for _format_contexts."""
    lossless_restatement = "Caroline finished a charity race."
    timestamp = "2023-05-03"
    location = None
    persons = ["Caroline"]
    entities = ["charity race"]
    topic = "charity race"


class FakeRetriever:
    def retrieve(self, query, enable_reflection=None):
        PaperFakeSystem.calls.append(("retrieve", query, enable_reflection))
        return [FakeEntry()]


class FakeAnswerGenerator:
    def _format_contexts(self, contexts):
        return "\n\n".join(
            f"[Context {i}] Content: {c.lossless_restatement}"
            for i, c in enumerate(contexts, 1))


class FakeLLMClient:
    def __init__(self, responses=None, fail_times=0):
        self.responses = list(responses or [])
        self.fail_times = fail_times

    def chat_completion(self, messages, temperature=0.2, response_format=None,
                        max_retries=None):
        PaperFakeSystem.calls.append(
            ("chat", temperature, max_retries, messages))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("api down")
        return self.responses.pop(0)

    def extract_json(self, text):
        return json.loads(text)


class PaperFakeSystem(FakeSystem):
    """FakeSystem with the components the paper-mode cat-5 path touches."""
    calls = []  # class-level: the worker builds the instance internally

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.hybrid_retriever = FakeRetriever()
        self.answer_generator = FakeAnswerGenerator()
        self.llm_client = FakeLLMClient(responses=[
            json.dumps({"reasoning": "not stated in the context",
                        "answer": R.ADVERSARIAL_REFUSAL})])


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


def test_paper_adversarial_happy_path(tmp):
    PaperFakeSystem.calls = []
    pred = tmp / "t6.jsonl"
    qa_rows = [
        (7, {"question": "Q7?", "category": 1, "answer": "A7", "evidence": []}),
        (30, dict(ADV_QA)),
    ]
    ci, n = R.run_one_locomo_conversation(
        0, ADV_ARGS_PAPER, {"conversation": FAKE_CONV}, qa_rows, tmp,
        PaperFakeSystem, set(), threading.Lock(), pred)
    assert (ci, n) == (0, 2), f"expected (0,2), got {(ci, n)}"
    rows = {r["sample_id"]: r for r in read(pred)}
    adv, normal = rows["locomo-0-qa-30"], rows["locomo-0-qa-7"]
    # category-5 row under the paper protocol
    assert adv["category"] == "adversarial"
    assert adv["protocol"] == R.ADVERSARIAL_PAPER_PROTOCOL
    assert adv["ground_truth"] == R.ADVERSARIAL_REFUSAL
    assert adv["prediction"] == R.ADVERSARIAL_REFUSAL
    assert adv["adversarial_answer"] == ADV_QA["adversarial_answer"]
    assert adv["error"] == ""
    # ordinary row untouched
    assert normal["protocol"] == ""
    assert normal["ground_truth"] == "A7"
    assert normal["prediction"].startswith("answer(")
    # retrieval ran with reflection disabled
    retrievals = [c for c in PaperFakeSystem.calls if c[0] == "retrieve"]
    assert len(retrievals) == 1, retrievals
    assert retrievals[0][1] == ADV_QA["question"]
    assert retrievals[0][2] is False, "reflection must be disabled for cat 5"
    # generation: temperature 0.5, max_retries 3, both options in the prompt
    chats = [c for c in PaperFakeSystem.calls if c[0] == "chat"]
    assert len(chats) == 1, chats
    assert chats[0][1] == 0.5 and chats[0][2] == 3, chats[0][1:3]
    user_msg = chats[0][3][1]["content"]
    assert "Option A:" in user_msg and "Option B:" in user_msg
    assert R.ADVERSARIAL_REFUSAL in user_msg
    assert ADV_QA["adversarial_answer"] in user_msg
    print("PASS test6 paper-mode cat5 row (fixed refusal ref, reflection off, "
          "binary-choice prompt, temp 0.5)")


def test_paper_adversarial_fallback(tmp):
    PaperFakeSystem.calls = []

    class FailingSystem(PaperFakeSystem):
        def __init__(self, *a, **k):
            FakeSystem.__init__(self, *a, **k)
            self.hybrid_retriever = FakeRetriever()
            self.answer_generator = FakeAnswerGenerator()
            self.llm_client = FakeLLMClient(fail_times=3)

    pred = tmp / "t7.jsonl"
    ci, n = R.run_one_locomo_conversation(
        0, ADV_ARGS_PAPER, {"conversation": FAKE_CONV}, [(30, dict(ADV_QA))],
        tmp, FailingSystem, set(), threading.Lock(), pred)
    assert (ci, n) == (0, 1)
    row = read(pred)[0]
    assert row["prediction"] == R.ADVERSARIAL_REFUSAL, row["prediction"]
    assert row["error"] == "", "generation fallback is not an error row"
    chats = [c for c in PaperFakeSystem.calls if c[0] == "chat"]
    assert len(chats) == 3, f"expected 3 attempts, got {len(chats)}"
    print("PASS test7 paper-mode cat5 fallback (3 attempts -> refusal string)")


def test_open_adversarial_mode(tmp):
    pred = tmp / "t8.jsonl"
    ci, n = R.run_one_locomo_conversation(
        0, ADV_ARGS_OPEN, {"conversation": FAKE_CONV}, [(30, dict(ADV_QA))],
        tmp, FakeSystem, set(), threading.Lock(), pred)
    assert (ci, n) == (0, 1)
    row = read(pred)[0]
    assert row["protocol"] == R.ADVERSARIAL_OPEN_PROTOCOL
    assert row["ground_truth"] == ""  # legacy behaviour: raw (empty) gold
    assert row["prediction"].startswith("answer(")
    print("PASS test8 open-mode cat5 row (legacy ask + empty gold + tag)")


def test_protocol_helpers():
    qa5, qa1 = dict(ADV_QA), {"category": 1, "answer": "A1"}
    assert R.adversarial_protocol_tag(ADV_ARGS_PAPER, qa5) == R.ADVERSARIAL_PAPER_PROTOCOL
    assert R.adversarial_protocol_tag(ADV_ARGS_OPEN, qa5) == R.ADVERSARIAL_OPEN_PROTOCOL
    assert R.adversarial_protocol_tag(ADV_ARGS_PAPER, qa1) == ""
    assert R.locomo_row_gold(ADV_ARGS_PAPER, qa5) == R.ADVERSARIAL_REFUSAL
    assert R.locomo_row_gold(ADV_ARGS_OPEN, qa5) == ""
    assert R.locomo_row_gold(ADV_ARGS_PAPER, qa1) == "A1"
    assert R.judge_template_key(
        {"category": "adversarial", "protocol": R.ADVERSARIAL_PAPER_PROTOCOL}
    ) == "adversarial-paper"
    assert R.judge_template_key(
        {"category": "adversarial", "protocol": R.ADVERSARIAL_OPEN_PROTOCOL}
    ) == "adversarial"
    assert R.judge_template_key({"category": "adversarial"}) == "adversarial"
    assert R.judge_template_key({"category": "single-hop"}) == "single-hop"
    assert R.judge_template_key({}) is None
    rendered = R.CATEGORY5_CHOICE_PROMPT.format(
        context_str="CTX", question="Q?", option_a="OA", option_b="OB")
    assert "Option A: OA" in rendered and "Option B: OB" in rendered
    assert R.ADVERSARIAL_REFUSAL in rendered
    print("PASS test9 protocol helpers + judge routing + prompt rendering")


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="locomo_qa_test_"))
    test_happy_path(tmp)
    test_resume_dedup(tmp)
    test_build_error_records(tmp)
    test_concurrent_workers(tmp)
    test_worker_exception_hardening(tmp)
    test_paper_adversarial_happy_path(tmp)
    test_paper_adversarial_fallback(tmp)
    test_open_adversarial_mode(tmp)
    test_protocol_helpers()
    print(f"ALL TESTS PASSED (tmp={tmp})")


if __name__ == "__main__":
    main()
