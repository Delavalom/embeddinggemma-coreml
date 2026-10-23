from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIXTURES = ROOT / "fixtures"
GATES = ROOT / "gates.toml"
HF_REPO = "google/embeddinggemma-2"

def stage_dir(stage: str) -> Path:
    d = ARTIFACTS / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
