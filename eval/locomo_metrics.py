"""
LoCoMo-standard lexical metrics: token F1 and BLEU.

- F1 uses the SQuAD convention (normalize -> tokenize -> overlap F1) used by the
  LoCoMo / mem0 / memobase evaluation suites.
- BLEU is self-contained (no nltk data needed) and reported in two flavours:
  * sentence BLEU with NLTK-equivalent method-1 (epsilon) smoothing, macro-averaged
    (this is the number mem0/memobase report);
  * standard corpus BLEU (clipped n-gram precision + brevity penalty, no smoothing).

Adversarial / Cognitive samples carry no usable gold answer and are therefore
excluded from F1 / BLEU aggregation by the caller.
"""
import re
import string
from collections import Counter
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Normalization (SQuAD / LoCoMo)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b")
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def normalize_answer(text) -> str:
    """Lower, strip punctuation/articles, collapse whitespace."""
    text = str(text if text is not None else "").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def tokenize(text) -> List[str]:
    return normalize_answer(text).split()


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def f1_score(prediction, ground_truth) -> Tuple[float, float, float]:
    """Return (F1, precision, recall) on normalized token multisets."""
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# ---------------------------------------------------------------------------
# BLEU (self-contained)
# ---------------------------------------------------------------------------

def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    c = Counter()
    for i in range(len(tokens) - n + 1):
        c[tuple(tokens[i:i + n])] += 1
    return c


def sentence_bleu(prediction, ground_truth, max_n: int = 4) -> float:
    """
    Sentence BLEU with NLTK method-1 (epsilon/add-one) smoothing and the
    standard brevity penalty. Works on normalized tokens.
    """
    pred = tokenize(prediction)
    ref = tokenize(ground_truth)
    if not pred:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngram_counts(pred, n)
        ref_ngrams = _ngram_counts(ref, n)
        clipped = sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        # NLTK ChenCherry method-1 style: add 1 to numerator and denominator
        precisions.append((clipped + 1.0) / (total + 1.0) if total > 0 else 0.0)

    if all(p == 0.0 for p in precisions):
        return 0.0

    brevity_penalty = 1.0 if len(pred) > len(ref) else pow(
        2.718281828459045, 1.0 - len(ref) / max(1, len(pred))
    )
    log_avg = sum(math_log(p) for p in precisions if p > 0) / max_n
    return brevity_penalty * math_exp(log_avg)


def corpus_bleu(predictions: Sequence, ground_truths: Sequence, max_n: int = 4) -> float:
    """Standard corpus BLEU: micro clipped precisions + brevity penalty."""
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len = 0
    ref_len = 0
    for prediction, ground_truth in zip(predictions, ground_truths):
        pred = tokenize(prediction)
        ref = tokenize(ground_truth)
        pred_len += len(pred)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            pred_ngrams = _ngram_counts(pred, n)
            ref_ngrams = _ngram_counts(ref, n)
            clipped[n - 1] += sum(min(cnt, ref_ngrams.get(ng, 0)) for ng, cnt in pred_ngrams.items())
            totals[n - 1] += sum(pred_ngrams.values())

    if pred_len == 0 or any(t == 0 for t in totals):
        return 0.0
    precisions = [clipped[i] / totals[i] for i in range(max_n)]
    if any(p == 0.0 for p in precisions):
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math_exp(
        1.0 - ref_len / pred_len
    )
    return brevity_penalty * math_exp(sum(math_log(p) for p in precisions) / max_n)


# natural log/exp kept as aliases so the file has no top-level import surprises
from math import log as math_log, exp as math_exp  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def has_gold(record: dict) -> bool:
    return bool(str(record.get("ground_truth") or "").strip())


def aggregate_lexical(records: List[dict]) -> Dict:
    """
    Aggregate F1 / BLEU over records that have a non-empty gold answer.
    Returns overall + per-category stats.
    """
    buckets: Dict[str, List[dict]] = {}
    for r in records:
        if not has_gold(r):
            continue
        buckets.setdefault(r.get("category") or "unknown", []).append(r)

    def _stats(rows: List[dict]) -> Dict:
        f1s, sbleus, sbleus_1, preds, golds = [], [], [], [], []
        for r in rows:
            pred, gold = r.get("prediction") or "", r.get("ground_truth") or ""
            f1s.append(f1_score(pred, gold)[0])
            sbleus.append(sentence_bleu(pred, gold))
            sbleus_1.append(sentence_bleu(pred, gold, max_n=1))
            preds.append(pred)
            golds.append(gold)
        n = len(rows)
        return {
            "n": n,
            "f1": round(sum(f1s) / n, 4) if n else None,
            "bleu_sentence": round(sum(sbleus) / n, 4) if n else None,
            "bleu_corpus": round(corpus_bleu(preds, golds), 4) if n else None,
            "bleu_1": round(sum(sbleus_1) / n, 4) if n else None,
            "bleu_corpus_1": round(corpus_bleu(preds, golds, max_n=1), 4) if n else None,
        }

    all_rows = [r for rows in buckets.values() for r in rows]
    return {
        "overall": _stats(all_rows),
        "by_category": {cat: _stats(rows) for cat, rows in sorted(buckets.items())},
    }
