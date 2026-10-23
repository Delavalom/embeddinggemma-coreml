from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIXTURES = ROOT / "fixtures"
GATES = ROOT / "gates.toml"

HF_REPO = "google/embeddinggemma-2"
# Commit of HF_REPO that every stage uses (checked 2026-10-22). Changing it makes s0 and everything after it stale.
HF_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
# sha256 of model.safetensors at HF_REVISION (its Git LFS oid). s0 checks the downloaded file against it.
HF_MODEL_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"

# Stage names in pipeline order. The justfile has the same list; tests/test_scripts.py keeps them in sync.
STAGES = (
    "s0_reference",
    "s1_rewrite",
    "s2_coreml_fp32",
    "s3_coreml_ane",
    "s4_compress",
    "s5_swift",
    "s6_device",
    "s7_publish",
)


def stage_dir(stage: str, root: Path = ROOT) -> Path:
    d = root / "artifacts" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
