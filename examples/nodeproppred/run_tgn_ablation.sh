#!/bin/bash
# TGN tgbn-trade ablation: quantify the TGB <-> tgm gap after the parity fixes.
#
# Two configurations, seeds 1-5, 50 epochs (TGB protocol: test evaluated every
# epoch, report test NDCG at the best-val epoch -- printed as "Best Test" at
# the end of each log):
#   strict : all bug fixes, tgm's strict temporal causality (t < label time)
#   parity : additionally --tgb-parity, replicating TGB's semantics (edges at
#            the label's own timestamp are ingested before prediction)
#
# parity - strict isolates the contribution of TGB's time semantics.
# The pre-fix baseline for comparison is PR #418: 0.37 val / 0.34 test.
# TGB leaderboard target: 0.395 +/- 0.002 val / 0.374 +/- 0.001 test.
#
# Usage: bash examples/nodeproppred/run_tgn_ablation.sh [outdir]

set -e
OUTDIR=${1:-tgn_ablation_logs}
mkdir -p "$OUTDIR"

for seed in 1 2 3 4 5; do
  echo "=== strict seed=$seed ==="
  python examples/nodeproppred/tgn.py \
    --dataset tgbn-trade --device cuda --epochs 50 --seed "$seed" \
    --log-file-path "$OUTDIR/strict_seed${seed}.log"

  echo "=== parity seed=$seed ==="
  python examples/nodeproppred/tgn.py \
    --dataset tgbn-trade --device cuda --epochs 50 --seed "$seed" --tgb-parity \
    --log-file-path "$OUTDIR/parity_seed${seed}.log"
done

echo
echo "=== Summary (test NDCG at best-val epoch) ==="
grep -H "Best Test" "$OUTDIR"/*.log
grep -H "Best Validation" "$OUTDIR"/*.log
