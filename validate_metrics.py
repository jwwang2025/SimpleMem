"""Quick validation: compute F1/BLEU + LLM-as-judge on the predictions already
produced by the smoke run, to prove the whole metric pipeline works end-to-end
before committing to the (multi-hour) full benchmark."""
import importlib.util
import json
import sys

ROOT = "/home/apulis-dev/code/SimpleMem"
sys.path.insert(0, ROOT)
sys.path.insert(0, f"{ROOT}/eval")

from locomo_metrics import aggregate_lexical  # noqa: E402
import simplemem.core.settings as s  # noqa: E402
from openai import OpenAI  # noqa: E402

# Load the eval driver module only for its judge templates/parser (no main() run).
spec = importlib.util.spec_from_file_location("rs", f"{ROOT}/eval/run_simplemem_eval.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

PREDS = f"{ROOT}/eval_output/smoke/predictions.jsonl"
rows = [json.loads(l) for l in open(PREDS, encoding="utf-8") if l.strip()]
print(f"Loaded {len(rows)} predictions")

# --- Lexical metrics (paper F1 + BLEU) -------------------------------
lex = aggregate_lexical(rows)
print("\n=== LEXICAL (gold-answer samples only) ===")
print(json.dumps(lex, ensure_ascii=False, indent=2))

# --- LLM-as-judge accuracy (Qwen3-8B as judge) ----------------------
client = OpenAI(base_url=s.settings.OPENAI_BASE_URL, api_key=s.settings.OPENAI_API_KEY)
model = "Qwen3-8B"
correct = partial = wrong = 0
print("\n=== JUDGE (Qwen3-8B) ===")
for r in rows:
    tk = rs.JUDGE_TEMPLATE_KEY.get(r["category"])
    tpl = rs.JUDGE_TEMPLATES.get(tk)
    if not tpl:
        print(f"  {r['sample_id']} ({r['category']}): no template, skipped")
        continue
    prompt = tpl.format(
        gold=str(r.get("ground_truth") or ""),
        pred=str(r.get("prediction") or ""),
        evidence=str(r.get("evidence") or ""),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    raw = (resp.choices[0].message.content or "").strip()
    label, reason = rs.parse_judge_response(raw)
    print(f"  {r['sample_id']} [{r['category']}] -> {label} | {reason[:80]}")
    if label == "correct":
        correct += 1
    elif label == "partial":
        partial += 1
    else:
        wrong += 1

n = correct + partial + wrong
if n:
    print("\n=== JUDGE SUMMARY ===")
    print(f"  strict accuracy (correct/n) : {correct / n:.3f}")
    print(f"  score avg (partial=0.5)     : {(correct + 0.5 * partial) / n:.3f}")
    print(f"  n judged                    : {n}")
