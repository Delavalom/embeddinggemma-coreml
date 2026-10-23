set shell := ["bash", "-euo", "pipefail", "-c"]
stages := "s0_reference s1_rewrite s2_coreml_fp32 s3_coreml_ane s4_compress s5_swift s6_device s7_publish"

setup:
    uv sync --group convert
    cd swift && swift package resolve

status:
    uv run python scripts/gen_status.py

stage n:
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(echo {{stages}} | tr ' ' '\n' | grep "^s{{n}}_")
    group=$(head -1 stages/$name/run.py | sed -n 's/^# group: //p')
    uv run --group "$group" python -m stages.$name.run
    uv run python scripts/gen_status.py >/dev/null

all:
    for n in 0 1 2 3 4 5; do just stage $n; done

test:
    uv run --group convert pytest -q
    cd swift && swift test

bench-device:
    just stage 6

publish *flags:
    uv run --group convert python -m stages.s7_publish.run {{flags}}
