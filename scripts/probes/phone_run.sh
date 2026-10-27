#!/bin/zsh
# usage: phone_run.sh TAG <probe args...>   e.g. phone_run.sh e1a-r1-A -fns embed_text_16
# Lists the app's E5 cache and the phone's free space, launches the probe, waits for its report, lists again.
# Writes ~/egcm-evidence-raw/<date>/<TAG>/{pre,post}.json, disk-{pre,post}.txt, probe.json, summary.json.
set -u
U=${UDID:-00008110-0014594C0A98401E}; APP=com.delavalom.egcm.demo; TAG=$1; shift
D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
list() { xcrun devicectl device info files --device $U --domain-type appDataContainer --domain-identifier $APP \
           --subdirectory Library --json-output $1 >/dev/null 2>&1; }
disk() { ideviceinfo -u $U -q com.apple.disk_usage 2>/dev/null | grep -v NANDInfo > $1; }
list $D/pre.json; disk $D/disk-pre.txt
date -u +%FT%TZ > $D/launched-at.txt
xcrun devicectl device process launch --device $U --terminate-existing $APP -- -tag $TAG "$@" >/dev/null 2>&1 \
  || { echo "$TAG LAUNCH FAILED (phone locked?)"; exit 1; }
for i in {1..240}; do   # up to 60 min
  sleep 15
  xcrun devicectl device copy from --device $U --domain-type appDataContainer --domain-identifier $APP \
    --source Documents/probe-$TAG.json --destination $D/probe.json >/dev/null 2>&1 || continue
  grep -q '"status":"running"' $D/probe.json || break
done
list $D/post.json; disk $D/disk-post.txt
cd $REPO && uv run --exact --group convert python -m scripts.probes.run_summary $D
