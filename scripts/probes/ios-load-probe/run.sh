#!/bin/zsh
# usage: run.sh TAG names  — launches the probe, streams new lines, stops at DONE, a gone process, or 15 min without progress
U=00008110-0014594C0A98401E; TAG=$1; NAMES=$2; F=/tmp/s6diag/probe-$TAG.txt; rm -f $F
xcrun devicectl device process launch --device $U --terminate-existing com.delavalom.egcm.demo -- -tag $TAG -hold "$NAMES" ${@:3} >/dev/null 2>&1 || { echo "LAUNCH FAILED $TAG"; exit 1; }
seen=0; last=$(date +%s)
while true; do
  xcrun devicectl device copy from --device $U --domain-type appDataContainer --domain-identifier com.delavalom.egcm.demo --source Documents/probe-$TAG.txt --destination $F >/dev/null 2>&1 || true
  n=$( [ -f $F ] && wc -l < $F | tr -d ' ' || echo 0 ); n=$((n+1)); [ -f $F ] || n=0
  if [ $n -gt $seen ]; then tail -n $((n-seen)) $F | grep -v "loading" ; seen=$n; last=$(date +%s); fi
  grep -q "DONE" $F 2>/dev/null && exit 0
  xcrun devicectl device info processes --device $U 2>/dev/null | grep -qi PhotoSearchDemo.app || { echo "$TAG PROCESS GONE"; exit 2; }
  [ $(( $(date +%s) - last )) -gt 900 ] && { echo "$TAG STALLED 15 min: $(tail -1 $F)"; exit 3; }
  sleep 15
done
