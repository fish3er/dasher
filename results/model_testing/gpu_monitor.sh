#!/bin/bash
# Co 2 s: temperatury, moc, obciążenie i zegary GPU + temp. CPU, dopóki działa proces $1.
# Format jak w poprzednich runach (gpu_monitor.csv), każda linia od razu na dysk.
PID=$1; OUT=$2
D=/sys/class/drm/card1/device; H=$D/hwmon/hwmon2; C=/sys/class/hwmon/hwmon3
echo "ts,edge_c,junction_c,mem_c,ppt_w,gpu_busy_pct,vram_mb,sclk_mhz,mclk_mhz,fan_rpm,vddgfx_mv,cpu_tctl_c" > "$OUT"
while kill -0 "$PID" 2>/dev/null; do
  sclk=$(grep '\*' $D/pp_dpm_sclk | sed -E 's/.*: ([0-9]+)Mhz.*/\1/')
  mclk=$(grep '\*' $D/pp_dpm_mclk | sed -E 's/.*: ([0-9]+)Mhz.*/\1/')
  echo "$(date +%Y-%m-%dT%H:%M:%S),$(( $(cat $H/temp1_input)/1000 )),$(( $(cat $H/temp2_input)/1000 )),$(( $(cat $H/temp3_input)/1000 )),$(awk '{printf "%.1f", $1/1e6}' $H/power1_average),$(cat $D/gpu_busy_percent),$(( $(cat $D/mem_info_vram_used)/1048576 )),$sclk,$mclk,$(cat $H/fan1_input),$(cat $H/in0_input),$(awk '{printf "%.1f", $1/1000}' $C/temp1_input)" >> "$OUT"
  sync "$OUT" 2>/dev/null
  sleep 2
done
