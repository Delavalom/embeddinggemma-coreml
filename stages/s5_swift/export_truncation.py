"""Token ids for texts longer than 128 tokens, truncated by the reference tokenizer (committed for the Swift tests).

    uv run --exact --group reference python -m stages.s5_swift.export_truncation --s0 <artifacts/s0_reference>

The reference tokenizer is transformers' `AutoTokenizer` on `tokenizer.json` / `tokenizer_config.json` at
`HF_REVISION` (the files `just test` caches in artifacts/tokenizer/<revision>/; downloaded if missing). Truncation is
Hugging Face's own: `tokenizer(prompt + text, truncation=True, max_length=128)`, which keeps `<bos>` and `<eos>`.
Before writing anything it checks that the same call (without truncation) reproduces s0's ids for all 200 fixture
texts, so this is the tokenizer path s0 used.

Writes swift/Tests/EmbeddingGemmaTests/Fixtures/truncation_ids.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egcm.fixtures import S0_DIR, load_texts
from egcm.paths import ARTIFACTS, HF_REPO, HF_REVISION, ROOT

OUT = ROOT / "swift/Tests/EmbeddingGemmaTests/Fixtures/truncation_ids.json"
MAX_TOKENS = 128
PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}
# Long inputs of different kinds: plain English, numbers and punctuation, code, mixed scripts and emoji, whitespace.
LONG_TEXTS = {
    "english": "The quick brown fox jumps over the lazy dog near the river bank. " * 30,
    "numbers": ", ".join(f"{i * 37 % 1000}.{i % 7}%" for i in range(400)),
    "code": "def f(x):\n    return [y * 2 for y in range(x) if y % 3 == 0]  # comment\n" * 25,
    "mixed": "Café au lait — 東京タワー 🗼 и Москва; ¿qué tal? العربية हिन्दी 👍🏽 " * 20,
    "spaces": "word   with    irregular\t\tspacing \n\n and  newlines " * 40,
}


def tokenizer_dir() -> Path:
    d = ARTIFACTS / "tokenizer" / HF_REVISION
    if not (d / "tokenizer.json").is_file():
        from huggingface_hub import hf_hub_download

        for f in ("tokenizer.json", "tokenizer_config.json"):
            hf_hub_download(HF_REPO, f, revision=HF_REVISION, local_dir=d)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s0", type=Path, default=S0_DIR)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir()))
    with np.load(a.s0 / "texts.npz") as z:
        for t in load_texts():
            ids = tok(PROMPTS[t["kind"]] + t["text"])["input_ids"]
            if ids != z[f"ids_{t['id']}"].tolist():
                raise AssertionError(f"reference tokenizer differs from s0 on {t['id']}")
    rows = []
    for name, text in LONG_TEXTS.items():
        for kind, prompt in PROMPTS.items():
            full = tok(prompt + text)["input_ids"]
            ids = tok(prompt + text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
            if len(full) <= MAX_TOKENS or len(ids) != MAX_TOKENS:
                raise AssertionError(f"{name}: {len(full)} tokens, truncated to {len(ids)}")
            rows.append({"id": f"{name}-{kind}", "kind": kind, "text": text, "full_tokens": len(full), "ids": ids})
    OUT.write_text(json.dumps({"hf_revision": HF_REVISION, "max_tokens": MAX_TOKENS, "texts": rows},
                              ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(rows)} texts; s0 ids reproduced for all fixtures)")


if __name__ == "__main__":
    main()
