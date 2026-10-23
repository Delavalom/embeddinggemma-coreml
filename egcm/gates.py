import tomllib
from egcm.paths import GATES

def gates(stage_key: str) -> dict:
    return tomllib.loads(GATES.read_text())[stage_key]

def ops_for(g: dict) -> dict[str, str]:
    return {k: ("<=" if k.startswith("max_") else "==" if isinstance(v, bool) else ">=") for k, v in g.items()}
