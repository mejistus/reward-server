"""
End-to-end smoke test for the MPS-friendly GenEval pipeline over a folder of
diffusion-generated images that don't ship with ground-truth prompts.

Strategy
--------
For each image we *bootstrap* the metadata from the model's own Mask2Former
detections:

* `single_object` — use the top-confidence detection's class name as the
  prompt (so the strict reward should be 1.0 if the test setup is sane).
* `counting`        — if the top class has >=2 instances, build a prompt
  asking for that exact count.
* `color_attr`      — for each image that contains >=2 distinct detected
  classes, ask for the top two classes and let CLIP decide on colors.

We then call `load_geneval().compute_geneval(...)` directly (no Flask server)
and print per-image results and aggregate statistics.

Run from the `reward-server/` directory:

    python test/test_geneval_mps.py
    python test/test_geneval_mps.py --image-dir ~/Pictures/diffusion
    python test/test_geneval_mps.py --device mps
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from glob import glob

import numpy as np
from PIL import Image


def _bootstrap_metadata(image_pil, seg, classnames, threshold=0.3):
    """Run Mask2Former once and synthesise a small metadata list from the
    detections. Mirrors the schema used by `compute_geneval`."""

    results = seg.infer([image_pil], score_threshold=0.0, mask_threshold=0.5)
    bboxes, masks = results[0]

    # Collect high-confidence detections, sorted desc by score.
    detections = []
    for cid, box_arr in enumerate(bboxes):
        for row in box_arr:
            score = float(row[4])
            if score >= threshold:
                detections.append((score, cid, classnames[cid]))
    detections.sort(key=lambda t: -t[0])

    if not detections:
        return [], detections

    top_score, top_cid, top_name = detections[0]
    # Group detections by class for counting-style metadata.
    per_class = defaultdict(list)
    for score, cid, name in detections:
        per_class[cid].append((score, name))

    metas = [
        {
            "tag": "single_object",
            "prompt": f"a photo of a {top_name}",
            "include": [{"class": top_name, "count": 1}],
        }
    ]

    # counting: ask for the exact count of the most-populated class (>=2).
    best_count_class = max(per_class.items(), key=lambda kv: len(kv[1]))
    best_cid, best_entries = best_count_class
    if len(best_entries) >= 2:
        metas.append({
            "tag": "counting",
            "prompt": f"a photo of {len(best_entries)} {best_entries[0][1]}s",
            "include": [{"class": best_entries[0][1], "count": len(best_entries)}],
        })

    # color_attr: top two distinct classes (counts>=1 each).
    seen_classes = []
    seen_cids = set()
    for score, cid, name in detections:
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        seen_classes.append(name)
        if len(seen_classes) == 2:
            break
    if len(seen_classes) == 2:
        metas.append({
            "tag": "color_attr",
            "prompt": f"a photo of a {seen_classes[0]} and a {seen_classes[1]}",
            "include": [
                {"class": seen_classes[0], "count": 1},
                {"class": seen_classes[1], "count": 1},
            ],
        })

    return metas, detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        default=os.path.expanduser("~/Pictures/diffusion"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many images (0 = no limit).")
    parser.add_argument("--print-detections", action="store_true",
                        help="Print top Mask2Former detections per image.")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from reward_server.gen_eval_mps import (
        HFSegmenter,
        _pick_device,
        load_geneval,
    )

    device = args.device if args.device != "auto" else _pick_device()
    print(f"[setup] device = {device}", flush=True)

    t0 = time.time()
    seg = HFSegmenter(device=device)
    print(f"[setup] mask2former loaded in {time.time()-t0:.2f}s", flush=True)

    t1 = time.time()
    fn = load_geneval(device=device)
    print(f"[setup] geneval pipeline loaded in {time.time()-t1:.2f}s", flush=True)

    with open(os.path.join(os.getcwd(), "reward_server/object_names.txt")) as f:
        classnames = [line.strip() for line in f]

    paths = sorted(
        p for p in glob(os.path.join(args.image_dir, "*"))
        if p.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if args.limit:
        paths = paths[: args.limit]
    print(f"[setup] found {len(paths)} images under {args.image_dir}", flush=True)

    overall = {
        "total_images": 0,
        "empty_images": 0,
        "by_tag": defaultdict(lambda: {"n": 0, "reward_sum": 0.0,
                                       "strict_sum": 0.0, "score_sum": 0.0}),
    }

    for path in paths:
        rel = os.path.basename(path)
        print(f"\n=== {rel} ===", flush=True)
        try:
            image_pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  failed to open: {e}")
            continue

        metas, detections = _bootstrap_metadata(image_pil, seg, classnames)
        if not metas:
            print("  no high-confidence detections; skipping.")
            overall["empty_images"] += 1
            continue
        if args.print_detections:
            for score, cid, name in detections[:8]:
                print(f"  detect: {name:<14s} cid={cid:>2d} score={score:.3f}")

        overall["total_images"] += 1
        # The MPS pipeline (matching the original mmdet-based one) pairs one
        # metadata entry with one image; we generate several metadata entries
        # per image, so we call the pipeline once per entry. To avoid paying
        # the Mask2Former cost repeatedly, we run a single detection batch
        # below and feed the resulting detections back through the evaluation
        # path. That keeps the call cheap.
        try:
            seg_results = seg.infer(
                [image_pil], score_threshold=0.0, mask_threshold=0.5
            )
        except Exception as e:
            print(f"  segmenter error: {e}")
            continue

        # Score / reward per metadata entry.
        per_entry_scores = []
        per_entry_rewards = []
        per_entry_strict = []
        for meta in metas:
            try:
                scores, rewards, strict_rewards, _g, _gs = fn(
                    [image_pil], [meta], only_strict=False
                )
                per_entry_scores.append(scores[0])
                per_entry_rewards.append(rewards[0])
                per_entry_strict.append(strict_rewards[0])
            except Exception as e:
                print(f"  pipeline error for tag={meta['tag']}: {e}")
                per_entry_scores.append(0.0)
                per_entry_rewards.append(0.0)
                per_entry_strict.append(0.0)

        for i, meta in enumerate(metas):
            tag = meta["tag"]
            bucket = overall["by_tag"][tag]
            bucket["n"] += 1
            bucket["reward_sum"] += float(per_entry_rewards[i])
            bucket["strict_sum"] += float(per_entry_strict[i])
            bucket["score_sum"] += float(per_entry_scores[i])
            print(
                f"  [{tag:<13s}] prompt={meta['prompt']!r}\n"
                f"      score={per_entry_scores[i]:.3f} "
                f"reward={per_entry_rewards[i]:.1f} "
                f"strict={per_entry_strict[i]:.1f}",
                flush=True,
            )

    print("\n=== summary ===", flush=True)
    for tag, b in sorted(overall["by_tag"].items()):
        if b["n"] == 0:
            continue
        print(
            f"  {tag:<13s} n={b['n']:>3d}  "
            f"avg_reward={b['reward_sum']/b['n']:.3f}  "
            f"avg_strict={b['strict_sum']/b['n']:.3f}  "
            f"avg_score={b['score_sum']/b['n']:.3f}"
        )
    print(
        f"\nprocessed={overall['total_images']}  "
        f"empty={overall['empty_images']}  "
        f"images_dir={args.image_dir}  "
        f"device={device}"
    )


if __name__ == "__main__":
    main()