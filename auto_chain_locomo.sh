#!/bin/bash
# auto_chain_locomo.sh — smoke health gate -> full LoCoMo-10 run.
# Lives OUTSIDE the git repo (operational script). Launch detached with setsid
# so it survives any SSH/session disconnect:
#   setsid nohup bash /home/apulis-dev/auto_chain_locomo.sh >/dev/null 2>&1 &
#
# Logic:
#   1. wait until eval_output/locomo_smoke/summary.json appears
#      (the harness writes it after the judge phase); bail out if the smoke
#      process dies without writing it
#   2. wait until no process with "out-dir eval_output/locomo_smoke" is alive
#      (account-level TPM budget is shared: never overlap two eval processes)
#   3. health gate: overall.n_429 == 0 AND overall.n_error == 0 AND
#      predictions >= 20 (2 convs x 5 cats x 2 QA)
#   4. launch the full run via run_locomo.sh (itself setsid-detached)
#   5. tail the full run every 30 min; on locomo_v1/summary.json, log the
#      final metrics and exit. Abort if the run process dies without summary.
REPO=/home/apulis-dev/code/SimpleMem
SMOKE_DIR=$REPO/eval_output/locomo_smoke
FULL_DIR=$REPO/eval_output/locomo_v1
LOG=/home/apulis-dev/auto_chain_locomo.log

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "=== auto-chain started (smoke gate -> full LoCoMo run) ==="

# ---------- phase 1: wait for smoke summary.json ----------
WAITED=0
while true; do
  if [ -f "$SMOKE_DIR/summary.json" ]; then
    say "smoke summary.json detected"
    break
  fi
  if ! pgrep -f "out-dir eval_output/locomo_smoke" >/dev/null 2>&1; then
    sleep 30  # grace period for the final flush
    if [ -f "$SMOKE_DIR/summary.json" ]; then
      say "smoke summary.json detected (just after process exit)"
      break
    fi
    say "FAIL: smoke process gone WITHOUT summary.json - full run NOT started"
    exit 1
  fi
  sleep 60
  WAITED=$((WAITED + 60))
  if [ "$WAITED" -ge 14400 ]; then
    say "FAIL: 4h timeout waiting for smoke - aborting chain"
    exit 1
  fi
done

# ---------- phase 2: make sure the smoke process is fully gone ----------
for i in $(seq 1 40); do
  pgrep -f "out-dir eval_output/locomo_smoke" >/dev/null 2>&1 || break
  sleep 15
done
if pgrep -f "out-dir eval_output/locomo_smoke" >/dev/null 2>&1; then
  say "WARN: smoke process still alive after 10min grace; proceeding anyway"
else
  say "smoke process confirmed exited"
fi

# ---------- phase 3: health gate ----------
read N429 NERR NPRED <<< "$(python3 - "$SMOKE_DIR" <<'PYEOF'
import json, os, sys
d = json.load(open(os.path.join(sys.argv[1], 'summary.json')))
ov = d.get('overall', {})
p = os.path.join(sys.argv[1], 'predictions.jsonl')
npred = sum(1 for _ in open(p)) if os.path.exists(p) else 0
print(ov.get('n_429', -1), ov.get('n_error', -1), npred)
PYEOF
)"
say "smoke health: n_429=$N429 n_error=$NERR n_predictions=$NPRED"

if [ "$N429" != "0" ]; then say "FAIL: n_429=$N429 - full run NOT started"; exit 1; fi
if [ "$NERR" != "0" ]; then say "FAIL: n_error=$NERR - full run NOT started"; exit 1; fi
if [ "$NPRED" -lt 20 ]; then say "FAIL: only $NPRED predictions (expected >=20) - full run NOT started"; exit 1; fi

# ---------- phase 4: launch the full run ----------
say "smoke HEALTHY - launching full LoCoMo run (10-way parallel, 1986 QA)"
cd "$REPO" || exit 1
bash run_locomo.sh >> "$LOG" 2>&1
say "full-run launcher returned; run continues detached"

# ---------- phase 5: tail the full run ----------
PRED=$FULL_DIR/predictions.jsonl
while true; do
  sleep 1800
  N=0
  [ -f "$PRED" ] && N=$(wc -l < "$PRED")
  say "full-run progress: $N/1986 predictions"
  if [ -f "$FULL_DIR/summary.json" ]; then
    say "=== FULL RUN COMPLETE ==="
    python3 - "$FULL_DIR" >> "$LOG" 2>&1 <<'PYEOF'
import json, os, sys
d = json.load(open(os.path.join(sys.argv[1], 'summary.json')))
ov = d['overall']
print('overall: n=%d judged=%d strict_acc=%.4f 429=%d err=%d llm_calls=%d' % (
    ov['n_predictions'], ov['n_judged'], ov['judge_accuracy_strict'],
    ov['n_429'], ov['n_error'], ov.get('n_llm_calls', -1)))
for cat, c in d.get('by_category', {}).items():
    print('  %-14s n=%-4d strict_acc=%.4f f1=%s bleu1=%s' % (
        cat, c['n_predictions'], c['judge_accuracy_strict'],
        c.get('f1'), c.get('bleu_1')))
PYEOF
    exit 0
  fi
  if ! pgrep -f "out-dir eval_output/locomo_v1" >/dev/null 2>&1; then
    say "FAIL: full-run process gone WITHOUT summary.json - aborting chain"
    exit 1
  fi
done
