# group: reference
"""s0_reference: golden vectors from Google's model, run the official way (sentence-transformers, FP32, CPU).

Writes to artifacts/s0_reference/: revision.txt, keys.json, texts.npz, images_70.npz, images_280.npz, debug.npz,
image_token_layout.json. See stages/s0_reference/README.md.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

from egcm.buckets import BUCKETS, BUDGETS, POOL, aspect_preserving_size, choose_bucket
from egcm.fixtures import load_images, load_texts, open_image
from egcm.paths import ARTIFACTS, HF_REPO, HF_REVISION, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s0_reference import reference as ref

STAGE = "s0_reference"
EMBEDDING_PATH = 'sentence_transformers.SentenceTransformer.encode({"image": img}, processing_kwargs=' \
                 '{"image": {"max_soft_tokens": budget}}); texts: encode(text, prompt_name="query"|"document")'
N_DETERMINISM = 10
MIN_PATH_COSINE = 0.99999
MAX_TEXT_TOKENS = 128  # the longest text function is embed_text_128; every fixture text must fit


def download() -> Path:
    snap = Path(snapshot_download(HF_REPO, revision=HF_REVISION, cache_dir=str(ARTIFACTS / "hf")))
    if snap.name != HF_REVISION:
        raise RuntimeError(f"snapshot {snap} is not revision {HF_REVISION}")
    return snap


def write_keys(snap: Path, out: Path) -> dict:
    with safe_open(snap / "model.safetensors", framework="pt") as f:
        keys = {k: [list(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()] for k in sorted(f.keys())}
    (out / "keys.json").write_text(json.dumps(keys, indent=1) + "\n")
    return keys


def card_license(snap: Path) -> str:
    front = (snap / "README.md").read_text().split("---")[1]
    return re.search(r"^license:\s*(\S+)", front, re.M).group(1)


def run_texts(m, texts: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embs, ids = [], {}
    for t in texts:
        prompt = "query" if t["kind"] == "query" else "document"
        ids[t["id"]] = ref.text_features(m, t["text"], prompt)["input_ids"][0].numpy().astype(np.int32)
        embs.append(ref.encode_text(m, t["text"], prompt))
    return np.stack(embs), ids


def run_image(m, img, budget: int) -> dict:
    """Both paths for one image at one budget: `raw` (the processor resizes) and `bucket` (we resize first)."""
    bucket = choose_bucket(img.height, img.width, budget)
    bimg, chw = ref.resize_to_bucket(img, bucket)
    feats = ref.image_features(m, bimg, budget)
    patches, pos = ref.valid_patches(feats)
    ids = feats["input_ids"][0].numpy().astype(np.int32)
    n_image = int((ids == ref.hf_model(m).config.image_token_id).sum())
    if len(patches) != bucket.tokens * POOL**2 or n_image != bucket.tokens:
        raise AssertionError(f"processor resized a {bucket.height}x{bucket.width} bucket image again "
                             f"({len(patches)} patches, expected {bucket.tokens * POOL**2})")
    return {"bucket": bucket, "emb_raw": ref.encode_image(m, img, budget),
            "emb_bucket": ref.encode_image(m, bimg, budget),
            "pixel_values": patches, "image_position_ids": pos, "input_ids": ids,
            "patchify_max_abs_diff": float(np.abs(patches - ref.patches_of(chw)).max())}


def run_images(m, images: list[dict], out: Path, log) -> dict[int, dict]:
    writers = {b: ref.NpzWriter(out / f"images_{b}.npz") for b in BUDGETS}
    res = {b: {"emb_raw": [], "emb_bucket": [], "bucket": [], "ids": {}, "first": [], "patch_diff": 0.0}
           for b in BUDGETS}
    for i, row in enumerate(images):
        img = open_image(row)
        for b in BUDGETS:
            r, w, acc = run_image(m, img, b), writers[b], res[b]
            for k in ("pixel_values", "image_position_ids", "input_ids"):
                w.add(f"{k}_{row['id']}", r[k])
            acc["emb_raw"].append(r["emb_raw"])
            acc["emb_bucket"].append(r["emb_bucket"])
            acc["bucket"].append(r["bucket"].name)
            acc["ids"][row["id"]] = r["input_ids"]
            acc["patch_diff"] = max(acc["patch_diff"], r["patchify_max_abs_diff"])
            if i < N_DETERMINISM:
                acc["first"].append(r)
        if i % 20 == 19:
            log(f"images: {i + 1}/{len(images)}")
    for b, w in writers.items():
        acc = res[b]
        acc["emb_raw"], acc["emb_bucket"] = np.stack(acc["emb_raw"]), np.stack(acc["emb_bucket"])
        w.add("order", np.array([r["id"] for r in images]))
        w.add("bucket", np.array(acc["bucket"]))
        w.add("height", np.array([r["height"] for r in images], dtype=np.int32))
        w.add("width", np.array([r["width"] for r in images], dtype=np.int32))
        w.add("raw_soft_tokens", np.array([_raw_tokens(r, b) for r in images], dtype=np.int32))
        w.add("emb_raw", acc["emb_raw"])
        w.add("emb_bucket", acc["emb_bucket"])
        w.close()
    return res


def _raw_tokens(row: dict, budget: int) -> int:
    h, w = aspect_preserving_size(row["height"], row["width"], budget)
    return (h // 48) * (w // 48)


def path_agreement(m, texts: list[dict], images: list[dict]) -> dict:
    """Cosine between `encode` and the manual path, on t000 and the first 2 images at both budgets."""
    cos = {}
    t = texts[0]
    cos[t["id"]] = float(ref.encode_text(m, t["text"], "query") @ ref.manual_embedding(
        m, ref.text_features(m, t["text"], "query")))
    for row in images[:2]:
        img = open_image(row)
        for b in BUDGETS:
            cos[f"{row['id']}@{b}"] = float(ref.encode_image(m, img, b) @ ref.manual_embedding(
                m, ref.image_features(m, img, b)))
    return cos


def write_debug(m, text: dict, image_row: dict, out: Path) -> list[str]:
    outs, ins = ref.debug_hooks(m)
    with ref.capture(outs, ins) as s:
        ref.encode_text(m, text["text"], "query")
    arrays = {f"text_{k}": v for k, v in s.items()}
    arrays["text_input_ids"] = ref.text_features(m, text["text"], "query")["input_ids"][0].numpy().astype(np.int32)
    img = open_image(image_row)
    bimg, _ = ref.resize_to_bucket(img, choose_bucket(img.height, img.width, 70))
    with ref.capture(outs, ins) as s:
        ref.encode_image(m, bimg, 70)
    arrays |= {f"image_{k}": v for k, v in s.items()}
    feats = ref.image_features(m, bimg, 70)
    arrays["image_input_ids"] = feats["input_ids"][0].numpy().astype(np.int32)
    arrays["image_pixel_values"], arrays["image_position_ids"] = ref.valid_patches(feats)
    np.savez(out / "debug.npz", **arrays)
    return sorted(arrays)


def write_layout(m, images: list[dict], res: dict, text_ids: np.ndarray, out: Path) -> dict:
    tok = ref.tokenizer(m)
    cfg = ref.hf_model(m).config
    special = {"pad": tok.pad_token_id, "bos": tok.bos_token_id, "eos": tok.eos_token_id,
               "boi": cfg.boi_token_id, "image": cfg.image_token_id, "eoi": cfg.eoi_token_id}
    buckets, patterns = {}, set()
    for bucket in BUCKETS:
        names = res[bucket.budget]["bucket"]
        i = names.index(bucket.name)
        ids = res[bucket.budget]["ids"][images[i]["id"]].tolist()
        expected = [special["bos"], special["boi"], *[special["image"]] * bucket.tokens, special["eoi"], special["eos"]]
        patterns.add(ids == expected)
        buckets[bucket.function_name] = {
            "fixture": images[i]["id"], "budget": bucket.budget, "bucket": bucket.name, "height": bucket.height,
            "width": bucket.width, "soft_tokens": bucket.tokens, "length": len(ids), "input_ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids)}
    layout = {
        "special_token_ids": special,
        "image_sequence": "<bos> <|image> <|image|> x soft_tokens <image|> <eos>"
                          if patterns == {True} else "DIFFERS per bucket: read input_ids",
        "matches_bos_boi_image_eoi_eos": patterns == {True},
        "text_example": {"fixture": "t000", "prompt_name": "query", "input_ids": text_ids.tolist(),
                         "tokens": tok.convert_ids_to_tokens(text_ids.tolist())},
        "buckets": buckets,
    }
    (out / "image_token_layout.json").write_text(json.dumps(layout, indent=1, ensure_ascii=False) + "\n")
    return layout


def check_determinism(m, texts, images, text_emb, text_ids, res) -> list[str]:
    """Rerun the first N_DETERMINISM texts and images and list every array that is not bitwise identical."""
    bad = []
    emb2, ids2 = run_texts(m, texts[:N_DETERMINISM])
    if not np.array_equal(emb2, text_emb[:N_DETERMINISM]):
        bad.append("text emb")
    bad += [f"text ids {k}" for k, v in ids2.items() if not np.array_equal(v, text_ids[k])]
    for i, row in enumerate(images[:N_DETERMINISM]):
        img = open_image(row)
        for b in BUDGETS:
            again, first = run_image(m, img, b), res[b]["first"][i]
            for k in ("emb_raw", "emb_bucket", "pixel_values", "image_position_ids", "input_ids"):
                if not np.array_equal(again[k], first[k]):
                    bad.append(f"{row['id']}@{b} {k}")
    return bad


def fill(report: Report) -> None:
    t_start = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s0 {time.monotonic() - t_start:7.1f}s] {msg}", flush=True)

    out = stage_dir(STAGE)
    snap = download()
    (out / "revision.txt").write_text(HF_REVISION + "\n")
    keys = write_keys(snap, out)
    prefixes: dict[str, int] = {}
    for k in keys:
        prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
    lic = card_license(snap)
    log(f"revision {HF_REVISION}, {len(keys)} keys, license {lic}")

    m = ref.load(snap)
    texts, images = load_texts(), load_images()
    text_emb, text_ids = run_texts(m, texts)
    np.savez(out / "texts.npz", emb=text_emb, order=np.array([t["id"] for t in texts]),
             **{f"ids_{k}": v for k, v in text_ids.items()})
    longest = max(len(v) for v in text_ids.values())
    if longest > MAX_TEXT_TOKENS:
        raise AssertionError(f"a fixture text has {longest} tokens; the longest text function takes {MAX_TEXT_TOKENS}")
    log(f"texts: {len(texts)} embedded, longest {longest} tokens")
    res = run_images(m, images, out, log)
    agreement = path_agreement(m, texts, images)
    debug_keys = write_debug(m, texts[0], images[0], out)
    layout = write_layout(m, images, res, text_ids["t000"], out)
    log("checking determinism")
    nondeterministic = check_determinism(m, texts, images, text_emb, text_ids, res)
    if min(agreement.values()) <= MIN_PATH_COSINE:
        raise AssertionError(f"encode and the manual path disagree: {agreement}")

    import sentence_transformers
    import torch
    import transformers

    report.metrics["license_apache"] = lic == "apache-2.0"
    report.metrics["key_inventory_written"] = (
        (out / "keys.json").is_file() and len(keys) > 0
        and all(prefixes.get(p, 0) > 0 for p in ("language_model", "vision_tower", "embed_vision")))
    report.metrics["deterministic"] = not nondeterministic
    report.metrics["path_agreement_min_cosine"] = min(agreement.values())
    report.details = {
        "revision": HF_REVISION, "license": lic, "embedding_path": EMBEDDING_PATH,
        "path_agreement_cosine": agreement, "nondeterministic": nondeterministic,
        "determinism_items": N_DETERMINISM, "key_count": len(keys), "key_prefixes": prefixes,
        "model_blob": (snap / "model.safetensors").resolve().name,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "sentence_transformers": sentence_transformers.__version__}, "torch_threads": ref.THREADS,
        "texts": len(texts), "images": len(images),
        "text_token_lengths": {"max": max(len(v) for v in text_ids.values()), "t199": len(text_ids["t199"])},
        "bucket_counts": {b: {n: res[b]["bucket"].count(n) for n in sorted(set(res[b]["bucket"]))} for b in BUDGETS},
        "patchify_max_abs_diff": {b: res[b]["patch_diff"] for b in BUDGETS},
        "image_sequence": layout["image_sequence"], "debug_arrays": debug_keys,
        "seconds": round(time.monotonic() - t_start, 1),
    }
    log(f"done: deterministic={not nondeterministic}, agreement min {min(agreement.values()):.8f}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
