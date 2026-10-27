#!/bin/zsh
# Builds the probe's bundled models (gitignored): b70 (the 9-function s4 package) and t16 (embed_text_16 alone, for
# loads without functionName). The probe compiles them on the device, like ModelStore does.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SRC=Examples/PhotoSearchDemo/Model/b70/EmbeddingGemma2.mlpackage
OUT=scripts/probes/ios-load-probe/ProbeModels; mkdir -p $OUT
rsync -a --delete $SRC/ $OUT/b70.mlpackage/
[ -d $OUT/t16.mlpackage ] || uv run --exact --group convert python scripts/probes/extract_functions.py $SRC $OUT/t16.mlpackage embed_text_16
du -sh $OUT/*
