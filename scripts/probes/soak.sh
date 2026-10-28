#!/bin/zsh
# usage: soak.sh PREFIX [COUNT] [GAP_MIN] [START]   the fix bar: launch 0 seeds the cache (cold expected), then COUNT launches
# (default 10) GAP_MIN minutes apart (default 165: launches 1..10 span 24.75 h), each loading all 9 b70 functions.
# Pass: every launch 1..COUNT warm. A launch that cannot start (phone locked or in use, Mac disconnected) is retried
# every 5 min and the delay is logged. Between launches the demo is closed and the keeper keeps the phone awake.
# Run detached: nohup caffeinate -is zsh scripts/probes/soak.sh soak1 > ~/egcm-evidence-raw/soak1.log 2>&1 &
set -u
cd "$(git -C "$(dirname $0)" rev-parse --show-toplevel)"
P=$1; N=${2:-10}; GAP=${3:-165}; S=${4:-0}   # START 1 skips the seed launch
FNS=embed_text_16,embed_text_32,embed_text_64,embed_text_128,embed_image_70_tall,embed_image_70_portrait,embed_image_70_square,embed_image_70_landscape,embed_image_70_wide
for n in $(seq $S $N); do
  due=$(date +%s); tries=0
  until zsh scripts/probes/phone_run.sh $P-$n -fns $FNS -retries 3 >/dev/null 2>&1; do
    tries=$((tries+1)); echo "$(date +%FT%T) $P-$n launch failed (try $tries), retrying in 5 min"; sleep 300
  done
  f=~/egcm-evidence-raw/$(date +%F)/$P-$n/probe.json
  python3 -c "
import json,sys;r=json.load(open('$f'));ls=[l for l in r['loads'] if l['fn'].startswith('embed')]
mx=max(l['ms'] for l in ls)/1000;bad=[l['fn'] for l in ls if not l['ok']]
print('$(date +%FT%T)','$P-$n','late_s',$(date +%s)-$due,'max_load_s',round(mx,1),'WARM' if mx<10 else 'COLD','fails',bad,'peakMB',max(l['peakMB'] for l in ls),'realFreeMB',r.get('realFreeMBStart'),'thermal',r['thermalStart'],r['thermalEnd'])"
  zsh scripts/probes/keep_awake.sh || echo "$(date +%FT%T) keeper launch failed"
  [ $n -lt $N ] && sleep $((GAP*60))
done
echo "$(date +%FT%T) soak $P finished"
