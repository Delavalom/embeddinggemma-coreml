"""usage: python -m scripts.probes.run_summary <run dir>   (written by phone_run.sh) → summary.json + one printed line."""
import json
import sys
from pathlib import Path

from scripts.probes.e5_listing import diff, load


def free_mb(path: Path) -> int | None:
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.startswith("AmountDataAvailable:"):
            return int(line.split()[1]) // 1_048_576
    return None


d = Path(sys.argv[1])
p = json.loads((d / "probe.json").read_text()) if (d / "probe.json").exists() else {"status": "no report", "loads": []}
pre, post = load(str(d / "pre.json")), load(str(d / "post.json"))
s = {"tag": d.name, "status": p["status"], "os": p.get("osBuild"), "since_last_s": p.get("secondsSinceLastProbe"),
     "thermal": [p.get("thermalStart"), p.get("thermalEnd")],
     "free_mb_app": [p.get("freeDiskMBStart"), p.get("freeDiskMBEnd")],
     "free_mb_mac": [free_mb(d / "disk-pre.txt"), free_mb(d / "disk-post.txt")],
     "avail_mem_mb": p.get("availMemMBStart"), "compile_ms": p.get("compileMS"),
     "loads": [(x["fn"], x["ok"], x["ms"], x["peakMB"], x.get("note") or (x.get("error") or "")[:100]) for x in p["loads"]],
     "cache_pre": pre["totals"], "cache_post": post["totals"], "cache_diff": diff(pre, post)}
(d / "summary.json").write_text(json.dumps(s, indent=1))
print(json.dumps({k: s[k] for k in ("tag", "status", "since_last_s", "free_mb_mac", "loads", "cache_post")}))
