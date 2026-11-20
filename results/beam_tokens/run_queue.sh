#!/bin/bash
# Kolejka beam_tokens (Gemma 4 E4B, podcast): sweep 15 konfiguracji na c_len 250 -> krzywa c_len 0–250
# -> ścieżki na 3 tokeny (beam po całości vs greedy). Kroki można podać jako argumenty, np. `paths`.
# Oba kroki są wznawialne: po twardym resecie PC wystarczy odpalić kolejkę ponownie.
# Bez limitu GPU (decyzja właściciela 2026-11-16).
# Uruchamiać z katalogu dasher-eval: setsid nohup results/beam_tokens/run_queue.sh [kroki] > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
BT=results/beam_tokens
QLOG=$BT/queue.log

log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; sync "$QLOG" 2>/dev/null; }

run() {  # $1 = tryb (sweep|curve|paths)
  local mode=$1 ts
  ts=$(date +%Y%m%d_%H%M%S)
  log "RUN $mode"
  python3 eval_tokens.py --mode "$mode" > "$BT/${mode}_$ts.log" 2>&1 &
  local pid=$!
  bash results/model_testing/gpu_monitor.sh $pid "$BT/gpu_monitor_${mode}_$ts.csv" &
  wait $pid; local rc=$?
  log "KONIEC $mode rc=$rc"
  return $rc
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(sweep curve paths)
log "START kolejki: ${STEPS[*]}"
for step in "${STEPS[@]}"; do run "$step"; done
log "KONIEC kolejki"
