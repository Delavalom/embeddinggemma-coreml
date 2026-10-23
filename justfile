set shell := ["bash", "-euo", "pipefail", "-c"]
stages := "s0_reference s1_rewrite s2_coreml_fp32 s3_coreml_ane s4_compress s5_swift s6_device s7_publish"

# Sync the convert environment and, once swift/ exists, resolve the Swift package.
# The reference environment is not synced here: `just stage 0` installs it on first use (uv run --exact).
setup:
    uv sync --group convert
    if [ -d swift ]; then (cd swift && swift package resolve); fi

status:
    uv run python -m scripts.gen_status

# Run stage N in the environment named by line 1 of its run.py (`# group: convert` or `# group: reference`).
# --exact makes the environment match that group alone, so reference-only packages (transformers) never
# leak into convert stages. STATUS.md is always regenerated, and the stage's exit code is kept.
stage n:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{n}}" in ''|*[!0-9]*) echo "error: stage must be a number, got '{{n}}'" >&2; exit 2;; esac
    name=$(printf '%s\n' {{stages}} | grep -x 's{{n}}_.*' || true)
    if [ -z "$name" ]; then
        echo "error: no stage numbered {{n}}. Stages: {{stages}}" >&2
        exit 2
    fi
    run="stages/$name/run.py"
    if [ ! -f "$run" ]; then
        echo "error: $run does not exist yet" >&2
        exit 2
    fi
    group=$(head -1 "$run" | sed -n 's/^# group: *//p')
    if [ -z "$group" ]; then
        echo "error: line 1 of $run must be '# group: convert' or '# group: reference'" >&2
        exit 2
    fi
    rc=0
    uv run --exact --group "$group" python -m "stages.$name.run" || rc=$?
    uv run --no-sync python -m scripts.gen_status >/dev/null || echo "warning: could not regenerate STATUS.md" >&2
    exit "$rc"

all:
    for n in 0 1 2 3 4 5; do just stage $n; done

test:
    uv run --exact --group convert pytest -q
    if [ -d swift ]; then (cd swift && swift test); fi

lint:
    uv run --group dev ruff check .

bench-device:
    just stage 6

publish *flags:
    uv run --exact --group convert python -m stages.s7_publish.run {{flags}}
