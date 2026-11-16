#!/bin/bash
# Kolejka runów model-testing: Bielik testing -> Bielik greedy -> Gemma greedy.
# Każdy run: pełny eval_v3_long (600 poz. x c_len 0/250/1000/2000/5000) na podcast_clean.txt.
# Logi i gpu_monitor ze znacznikiem czasu, żeby nie nadpisać poprzednich (np. przerwanego Gemma greedy).
# Uruchamiać z katalogu dasher-eval: setsid nohup results/model_testing/run_queue.sh &
set -u
cd "$(dirname "$0")/../.." || exit 1
MT=results/model_testing
QLOG=$MT/queue.log
BIELIK=models/Bielik-11B-v3.1-Instruct.Q5_K_M.gguf
GEMMA=models/google_gemma-4-E4B-it-Q4_K_M.gguf

log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; sync "$QLOG" 2>/dev/null; }

perf=$(cat /sys/class/drm/card1/device/power_dpm_force_performance_level)
cap=$(cat /sys/class/drm/card1/device/hwmon/hwmon2/power1_cap)
log "START kolejki; perf_level=$perf power1_cap=$((cap/1000000)) W"

run() {  # $1 = nazwa katalogu, $2 = --search, $3 = gguf
  local name=$1 search=$2 gguf=$3 ts out
  ts=$(date +%Y%m%d_%H%M%S); out=$MT/$name; mkdir -p "$out"
  log "RUN $name ($search, $(basename "$gguf"))"
  python3 eval_context.py --config configs/eval_v3_long.yaml \
    --corpus-dir corpus_context_podcast_pl --corpus-glob podcast_clean.txt \
    --search "$search" --gguf "$gguf" --results-dir "$out" \
    > "$out/run_$ts.log" 2>&1 &
  local pid=$!
  bash $MT/gpu_monitor.sh $pid "$out/gpu_monitor_$ts.csv" &
  wait $pid; local rc=$?
  log "KONIEC $name rc=$rc"
  return $rc
}

run bielik_testing      testing        $BIELIK
run bielik_greedy       testing_greedy $BIELIK
run gemma-4-E4B_greedy  testing_greedy $GEMMA
log "KONIEC kolejki"
