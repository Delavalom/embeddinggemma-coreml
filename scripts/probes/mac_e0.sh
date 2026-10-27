#!/bin/zsh
# usage: mac_e0.sh TAG MODEL_DIR FN [units]   e.g. mac_e0.sh e0-p1 /tmp/e0/b70.mlmodelc embed_text_16 ne
# One probe process, one function; records the Mac E5 cache before/after and the model files' inode/mtime.
set -eu
TAG=$1; M=$2; FN=$3; UNITS=${4:-ne}; D=~/egcm-evidence-raw/$(date +%F)/$TAG; mkdir -p $D
C=~/Library/Caches/probe/com.apple.e5rt.e5bundlecache
REPO=$(git -C "$(dirname $0)" rev-parse --show-toplevel)
[ -x /tmp/probe ] || swiftc -O $REPO/scripts/probes/mac_load_probe.swift -o /tmp/probe
cd $REPO
e5() { uv run --exact --group convert python -m scripts.probes.e5_listing "$@"; }
e5 summarize $C > $D/pre.json
stat -f '%i %m %z %N' $M/model.mil $M/weights/weight.bin > $D/model-stat.txt
/tmp/probe $M $UNITS $FN > $D/probe.txt 2>&1
e5 summarize $C > $D/post.json
uv run --exact --group convert python -c "import json,sys; from scripts.probes.e5_listing import diff; a,b=(json.load(open(f)) for f in sys.argv[1:3]); d=diff(a,b); print(len(d['added_keys']), 'keys added;', len(d['payload_lost']), 'payload lost')" $D/pre.json $D/post.json > $D/diff.txt
echo "$TAG $(grep loaded $D/probe.txt | awk '{print $2, $3, $5}') | $(cat $D/diff.txt)"
