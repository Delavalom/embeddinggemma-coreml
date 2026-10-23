# group: reference
"""s0_reference: golden vectors from Google's model, run the official way (sentence-transformers, FP32, CPU).

Writes to artifacts/s0_reference/: revision.txt, keys.json, texts.npz, images_70.npz, images_280.npz, debug.npz,
image_token_layout.json. See stages/s0_reference/README.md.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

from egcm.buckets import BUCKETS, BUDGETS, POOL, choose_bucket
from egcm.fixtures import load_images, load_texts, open_image, sha256_file, verify_manifest
from egcm.paths import ARTIFACTS, HF_MODEL_SHA256, HF_REPO, HF_REVISION, stage_dir
from egcm.report import Report
from egcm.stage import run_stage
from stages.s0_reference import artifacts, pixels
from stages.s0_reference import reference as ref

STAGE = "s0_reference"
EMBEDDING_PATH = 'sentence_transformers.SentenceTransformer.encode({"image": img}, processing_kwargs=' \
                 '{"image": {"max_soft_tokens": budget}}); texts: encode(text, prompt_name="query"|"document")'
MIN_PATH_COSINE = 0.99999
MAX_TEXT_TOKENS = 128  # the longest text function is embed_text_128; every fixture text must fit


@dataclass
class BudgetResult:
    """Everything s0 keeps in memory for one budget (pixel arrays go straight to disk)."""

    budget: int
    order: list[str] = field(default_factory=list)
    bucket: list[str] = field(default_factory=list)
    emb_raw: list[np.ndarray] = field(default_factory=list)
    emb_bucket: list[np.ndarray] = field(default_factory=list)
    raw_soft_tokens: list[int] = field(default_factory=list)
    ids: dict[str, np.ndarray] = field(default_factory=dict)
    sample: dict[str, dict] = field(default_factory=dict)  # first-pass results of the determinism sample

    def raw_vs_bucket(self) -> dict[str, float]:
        cos = np.einsum("nd,nd->n", np.stack(self.emb_raw), np.stack(self.emb_bucket))
        return {"min": float(cos.min()), "median": float(np.median(cos))}


def download() -> Path:
    snap = Path(snapshot_download(HF_REPO, revision=HF_REVISION, cache_dir=str(ARTIFACTS / "hf")))
    if snap.name != HF_REVISION:
        raise RuntimeError(f"snapshot {snap} is not revision {HF_REVISION}")
    digest = sha256_file(snap / "model.safetensors")
    if digest != HF_MODEL_SHA256:
        raise RuntimeError(f"model.safetensors has sha256 {digest}, expected {HF_MODEL_SHA256}")
    return snap


def tensor_inventory(snap: Path) -> dict[str, tuple[list[int], str]]:
    with safe_open(snap / "model.safetensors", framework="pt") as f:
        return {k: (list(f.get_slice(k).get_shape()), f.get_slice(k).get_dtype()) for k in f.keys()}


def card_license(snap: Path) -> str:
    front = (snap / "README.md").read_text().split("---")[1]
    return re.search(r"^license:\s*(\S+)", front, re.M).group(1)


def run_texts(m, texts: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embs, ids = [], {}
    for t in texts:
        name = ref.prompt_name(t["kind"])
        ids[t["id"]] = ref.text_features(m, t["text"], name)["input_ids"][0].numpy().astype(np.int32)
        embs.append(ref.encode_text(m, t["text"], name))
    return np.stack(embs), ids


def check_pixel_contract(patches, pos, u8, bucket, ids, image_token_id) -> None:
    """The processor saw exactly our bucket image: no second resize, patches equal to `u8 * f32(1/255)` in
    upstream patch order (bit for bit), positions equal to the bucket's meshgrid."""
    n_image = int((ids == image_token_id).sum())
    if len(patches) != bucket.tokens * POOL**2 or n_image != bucket.tokens:
        raise AssertionError(f"processor resized a {bucket.height}x{bucket.width} bucket image again "
                             f"({len(patches)} patches, {n_image} image tokens, expected {bucket.tokens})")
    if not np.array_equal(patches, pixels.patchify(u8)):
        raise AssertionError(f"processor patches differ from patchify(u8) for {bucket.function_name}: "
                             f"max abs diff {np.abs(patches - pixels.patchify(u8)).max()}")
    if not np.array_equal(pos, pixels.patch_positions(bucket.height, bucket.width)):
        raise AssertionError(f"image_position_ids are not the {bucket.function_name} meshgrid")


def run_image(m, img, budget: int) -> dict:
    """Both paths for one image at one budget: `raw` (the processor resizes) and `bucket` (we resize first)."""
    bucket = choose_bucket(img.height, img.width, budget)
    bimg, u8 = ref.resize_to_bucket(img, bucket)
    feats = ref.image_features(m, bimg, budget)
    patches, pos = ref.valid_patches(feats)
    ids = feats["input_ids"][0].numpy().astype(np.int32)
    check_pixel_contract(patches, pos, u8, bucket, ids, ref.hf_model(m).config.image_token_id)
    return {"bucket": bucket, "emb_raw": ref.encode_image(m, img, budget),
            "emb_bucket": ref.encode_image(m, bimg, budget),
            "raw_soft_tokens": ref.processor_soft_tokens(m, img, budget),
            "pixels": u8, "image_position_ids": pos, "input_ids": ids}


def run_images(m, images: list[dict], sample: set[str], out: Path, log) -> dict[int, BudgetResult]:
    res = {b: BudgetResult(b) for b in BUDGETS}
    with contextlib.ExitStack() as stack:
        writers = {b: stack.enter_context(artifacts.NpzWriter(out / f"images_{b}.npz")) for b in BUDGETS}
        for i, row in enumerate(images):
            img = open_image(row)
            for b in BUDGETS:
                r, w, acc = run_image(m, img, b), writers[b], res[b]
                for k in ("pixels", "image_position_ids", "input_ids"):
                    w.add(f"{k}_{row['id']}", r[k])
                acc.order.append(row["id"])
                acc.bucket.append(r["bucket"].name)
                acc.emb_raw.append(r["emb_raw"])
                acc.emb_bucket.append(r["emb_bucket"])
                acc.raw_soft_tokens.append(r["raw_soft_tokens"])
                acc.ids[row["id"]] = r["input_ids"]
                if row["id"] in sample:
                    acc.sample[row["id"]] = r
            if i % 20 == 19:
                log(f"images: {i + 1}/{len(images)}")
        for b, w in writers.items():
            acc = res[b]
            w.add("order", np.array(acc.order))
            w.add("bucket", np.array(acc.bucket))
            w.add("height", np.array([r["height"] for r in images], dtype=np.int32))
            w.add("width", np.array([r["width"] for r in images], dtype=np.int32))
            w.add("raw_soft_tokens", np.array(acc.raw_soft_tokens, dtype=np.int32))
            w.add("emb_raw", np.stack(acc.emb_raw))
            w.add("emb_bucket", np.stack(acc.emb_bucket))
    return res


def path_agreement(m, texts: list[dict], images: list[dict]) -> dict[str, float]:
    """Cosine between `encode` and the manual path; raises below MIN_PATH_COSINE."""
    cos = {}
    for t in texts:
        name = ref.prompt_name(t["kind"])
        cos[t["id"]] = float(ref.encode_text(m, t["text"], name)
                             @ ref.manual_embedding(m, ref.text_features(m, t["text"], name)))
    for row in images:
        img = open_image(row)
        for b in BUDGETS:
            cos[f"{row['id']}@{b}"] = float(ref.encode_image(m, img, b)
                                            @ ref.manual_embedding(m, ref.image_features(m, img, b)))
    if min(cos.values()) <= MIN_PATH_COSINE:
        raise AssertionError(f"encode and the manual path disagree: {cos}")
    return cos


def write_debug(m, text: dict, image_row: dict, out: Path) -> list[str]:
    outs, ins = ref.debug_hooks(m)
    name = ref.prompt_name(text["kind"])
    with ref.capture(outs, ins) as s:
        ref.encode_text(m, text["text"], name)
    arrays = {f"text_{k}": v for k, v in s.items()}
    arrays["text_input_ids"] = ref.text_features(m, text["text"], name)["input_ids"][0].numpy().astype(np.int32)
    img = open_image(image_row)
    bimg, u8 = ref.resize_to_bucket(img, choose_bucket(img.height, img.width, 70))
    with ref.capture(outs, ins) as s:
        ref.encode_image(m, bimg, 70)
    arrays |= {f"image_{k}": v for k, v in s.items()}
    feats = ref.image_features(m, bimg, 70)
    arrays["image_input_ids"] = feats["input_ids"][0].numpy().astype(np.int32)
    arrays["image_pixels"] = u8
    arrays["image_position_ids"] = ref.valid_patches(feats)[1]
    with artifacts.NpzWriter(out / "debug.npz") as w:
        for k, v in arrays.items():
            w.add(k, v)
    return sorted(arrays)


def write_layout(m, images: list[dict], res: dict[int, BudgetResult], text: dict, text_ids, out: Path) -> dict:
    tok, cfg = ref.tokenizer(m), ref.hf_model(m).config
    special = {"pad": tok.pad_token_id, "bos": tok.bos_token_id, "eos": tok.eos_token_id,
               "boi": cfg.boi_token_id, "image": cfg.image_token_id, "eoi": cfg.eoi_token_id}
    buckets = {}
    for bucket in BUCKETS:
        acc = res[bucket.budget]
        i = acc.bucket.index(bucket.name)
        ids = acc.ids[acc.order[i]].tolist()
        buckets[bucket.function_name] = {
            "fixture": acc.order[i], "budget": bucket.budget, "bucket": bucket.name, "height": bucket.height,
            "width": bucket.width, "soft_tokens": bucket.tokens, "length": len(ids), "input_ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids)}
    example = {"fixture": text["id"], "prompt_name": ref.prompt_name(text["kind"]), "input_ids": text_ids.tolist(),
               "tokens": tok.convert_ids_to_tokens(text_ids.tolist())}
    return artifacts.write_layout(out, special, buckets, example)


def check_determinism(m, texts: list[dict], sample_texts: list[dict], images: list[dict], text_emb, text_ids,
                      res: dict[int, BudgetResult]) -> list[str]:
    """Rerun the sample in-process and list every array that is not bitwise identical to the first pass."""
    bad = []
    index = {t["id"]: i for i, t in enumerate(texts)}
    emb2, ids2 = run_texts(m, sample_texts)
    for k, e in zip(ids2, emb2, strict=True):
        if not np.array_equal(e, text_emb[index[k]]) or not np.array_equal(ids2[k], text_ids[k]):
            bad.append(f"text {k}")
    for row in images:
        if row["id"] not in res[BUDGETS[0]].sample:
            continue
        img = open_image(row)
        for b in BUDGETS:
            again, first = run_image(m, img, b), res[b].sample[row["id"]]
            for k in ("emb_raw", "emb_bucket", "pixels", "image_position_ids", "input_ids", "raw_soft_tokens"):
                if not np.array_equal(again[k], first[k]):
                    bad.append(f"{row['id']}@{b} {k}")
    return bad


def samples(texts: list[dict], images: list[dict], text_ids: dict) -> tuple[list[dict], list[dict]]:
    """The determinism and path-agreement sample: the first query, the first document, the longest text, and the
    first image of each bucket shape."""
    longest = max(texts, key=lambda t: len(text_ids[t["id"]]))
    picked_t = {next(t["id"] for t in texts if t["kind"] == k) for k in ("query", "document")} | {longest["id"]}
    shapes: dict[str, str] = {}
    for r in images:
        shapes.setdefault(choose_bucket(r["height"], r["width"], BUDGETS[0]).name, r["id"])
    return [t for t in texts if t["id"] in picked_t], [r for r in images if r["id"] in set(shapes.values())]


def fill(report: Report) -> None:
    t_start = time.monotonic()

    def log(msg: str) -> None:
        print(f"[s0 {time.monotonic() - t_start:7.1f}s] {msg}", flush=True)

    if problems := verify_manifest():
        raise RuntimeError(f"fixtures do not match MANIFEST.sha256: {problems[:10]}")
    out = stage_dir(STAGE)
    snap = download()
    (out / "revision.txt").write_text(HF_REVISION + "\n")
    tensors = tensor_inventory(snap)
    artifacts.write_keys(tensors, out)
    prefixes: dict[str, int] = {}
    for k in tensors:
        prefixes[k.split(".")[0]] = prefixes.get(k.split(".")[0], 0) + 1
    lic = card_license(snap)
    log(f"revision {HF_REVISION}, {len(tensors)} keys, license {lic}")

    m = ref.load(snap)
    texts, images = load_texts(), load_images()
    text_emb, text_ids = run_texts(m, texts)
    longest = max(texts, key=lambda t: len(text_ids[t["id"]]))
    if len(text_ids[longest["id"]]) > MAX_TEXT_TOKENS:
        raise AssertionError(f"{longest['id']} has {len(text_ids[longest['id']])} tokens; "
                             f"the longest text function takes {MAX_TEXT_TOKENS}")
    sample_texts, sample_images = samples(texts, images, text_ids)
    agreement = path_agreement(m, sample_texts, images[:2])
    log(f"texts: {len(texts)} embedded, longest {longest['id']} ({len(text_ids[longest['id']])} tokens); "
        f"path agreement min {min(agreement.values()):.8f}")
    artifacts.write_texts(out, [t["id"] for t in texts], text_emb, text_ids)

    res = run_images(m, images, {r["id"] for r in sample_images}, out, log)
    debug_keys = write_debug(m, texts[0], images[0], out)
    layout = write_layout(m, images, res, texts[0], text_ids[texts[0]["id"]], out)
    log("checking determinism")
    nondeterministic = check_determinism(m, texts, sample_texts, images, text_emb, text_ids, res)

    import sentence_transformers
    import torch
    import transformers

    report.metrics["license_apache"] = lic == "apache-2.0"
    report.metrics["key_inventory_written"] = (
        (out / "keys.json").is_file() and len(tensors) > 0
        and all(prefixes.get(p, 0) > 0 for p in ("language_model", "vision_tower", "embed_vision")))
    report.metrics["deterministic"] = not nondeterministic
    report.metrics["path_agreement_min_cosine"] = min(agreement.values())
    report.details = {
        "revision": HF_REVISION, "model_sha256": HF_MODEL_SHA256, "license": lic, "embedding_path": EMBEDDING_PATH,
        "path_agreement_cosine": agreement, "nondeterministic": nondeterministic,
        "determinism_sample": [t["id"] for t in sample_texts] + [r["id"] for r in sample_images],
        "key_count": len(tensors), "key_prefixes": prefixes,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__,
                     "sentence_transformers": sentence_transformers.__version__}, "torch_threads": ref.THREADS,
        "texts": len(texts), "images": len(images),
        "longest_text": {"id": longest["id"], "tokens": len(text_ids[longest["id"]])},
        "bucket_counts": {b: {n: res[b].bucket.count(n) for n in sorted(set(res[b].bucket))} for b in BUDGETS},
        "raw_vs_bucket_cosine": {b: res[b].raw_vs_bucket() for b in BUDGETS},
        "image_sequence": layout["image_sequence"], "debug_arrays": debug_keys,
        "seconds": round(time.monotonic() - t_start, 1),
    }
    log(f"done: deterministic={not nondeterministic}, raw_vs_bucket {report.details['raw_vs_bucket_cosine']}")


def main() -> None:
    run_stage(STAGE, fill)


if __name__ == "__main__":
    main()
