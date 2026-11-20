#!/bin/bash
# Większa próbka: beam w5_k5_p1 vs greedy (i greedy_top5) na pasku 3-tokenowym, żeby rozstrzygnąć Hit@1.
# Zbiór: wszystkie początki zdania (466) + 1000 kolejnych słów + 1000 cięć w połowie słowa, seed 2.
# Wznawialny. Uruchamiać z katalogu dasher-eval: setsid nohup results/beam_tokens/run_large.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
BT=results/beam_tokens
ts=$(date +%Y%m%d_%H%M%S)
echo "$(date '+%F %T') RUN paths_large" >> $BT/queue.log
python3 eval_tokens.py --mode paths --testset $BT/testset_podcast_large.jsonl \
  --configs w1_k1_p1,greedy_top5,w5_k5_p1 --out $BT/paths_gemma_large > $BT/paths_large_$ts.log 2>&1 &
pid=$!
bash results/model_testing/gpu_monitor.sh $pid "$BT/gpu_monitor_paths_large_$ts.csv" &
wait $pid
echo "$(date '+%F %T') KONIEC paths_large rc=$?" >> $BT/queue.log
