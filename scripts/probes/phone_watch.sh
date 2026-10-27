#!/bin/zsh
# usage: phone_watch.sh TAG MINUTES COUNT   lists the E5 cache and free space COUNT times, MINUTES apart, launching nothing
U=${UDID:-00008110-0014594C0A98401E}; TAG=$1; D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
for i in $(seq 1 $3); do
  f=$D/watch-$(date +%H%M%S).json
  xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier com.delavalom.egcm.demo \
    --subdirectory Library --json-output $f >/dev/null 2>&1
  free=$(ideviceinfo -u $U -q com.apple.disk_usage 2>/dev/null | awk '/^AmountDataAvailable/{printf "%.0f", $2/1048576}')
  (cd $REPO && uv run --exact --group convert python -c "import json,sys; from scripts.probes.e5_listing import load; print(sys.argv[1][-11:-5], 'free_mb', sys.argv[2], json.dumps(load(sys.argv[1])['totals']))" $f $free)
  if [ $i -lt $3 ]; then sleep $(( $2 * 60 )); fi
done
