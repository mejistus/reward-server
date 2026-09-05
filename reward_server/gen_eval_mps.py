"""
GenEval adapted for Apple Silicon (MPS) backend.

Drops the entire `mmdet` / `mmcv` / `mmengine` dependency stack and replaces
the original mmdet-based Mask2Former inference with the HuggingFace
`transformers` Mask2Former checkpoint (facebook/mask2former-swin-small-coco-instance).
OpenAI CLIP (via `open_clip`) is kept for zero-shot color classification.

Public API matches the original `gen_eval.load_geneval()` so callers
(`app_geneval.py`, `test/test_geneval.py`) work without modification.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc

zsc.tqdm = lambda it, *args, **kwargs: it


def _pick_device(prefer: str = "mps") -> str:
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = _pick_device("mps")

# HuggingFace model id of the Mask2Former variant the original repo used
# (mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco, instance segmentation head,
# 80 COCO stuff/thing categories). Equivalent to the mmdet checkpoint the
# original code downloads.
HF_M2F_REPO = os.environ.get(
    "GENEVAL_M2F_REPO", "facebook/mask2former-swin-small-coco-instance"
)

# Cache directory for HuggingFace / open_clip downloads (overridable via env).
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return [x1, y1, x2, y2] (inclusive) for a boolean HxW mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
    )


class HFSegmenter:
    """Thin wrapper around HF Mask2Former that mimics the slice of the
    mmdet `inference_detector` API we actually use: per-image instance
    segmentation producing per-class bboxes and binary masks.

    We replicate HF's score = softmax(class_logits) * mask_avg_prob math
    inline (see
    `transformers.models.mask2former.image_processing_mask2former.post_process_instance_segmentation`)
    so we can return *every* query and let the caller filter by its own
    threshold, instead of being constrained to HF's hard-coded
    `threshold=0.5` default.
    """

    NUM_QUERIES = 100  # matches facebook/mask2former-swin-small-coco-instance

    def __init__(self, repo_id: str = HF_M2F_REPO, device: str = DEVICE):
        self.device = device
        self.processor = Mask2FormerImageProcessor.from_pretrained(repo_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            repo_id
        ).to(device).eval()
        cfg = self.model.config
        self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        self.num_classes = len(self.id2label)

    @torch.no_grad()
    def infer(
        self,
        images: list,
        score_threshold: float = 0.0,
        mask_threshold: float = 0.5,
    ) -> list:
        """Run instance segmentation on a batch of PIL images.

        Returns a list (one per input image) where each entry is a tuple
        `(bboxes, masks)`:
            - `bboxes`: list of length num_classes; each element is an
              (N_i, 5) float array of [x1, y1, x2, y2, score].
            - `masks`:  list of length num_classes; each element is either
              None (if no instances of that class) or a (N_i, H, W) uint8
              array of binary masks.
        """
        np_imgs = [np.array(im.convert("RGB")) for im in images]
        target_sizes = [im.shape[:2] for im in np_imgs]

        inputs = self.processor(images=np_imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        class_logits = outputs.class_queries_logits  # (B, Q, num_classes+1)
        mask_logits = outputs.masks_queries_logits  # (B, Q, H/4, W/4)

        results = []
        for b in range(class_logits.shape[0]):
            cls = class_logits[b]                           # (Q, C+1)
            msk = mask_logits[b]                            # (Q, h, w)

            # Replicate HF's score formula but keep every query.
            scores = torch.softmax(cls, dim=-1)[:, :-1]     # (Q, C)
            scores_per_query, labels_per_query = scores.max(dim=-1)  # (Q,)

            # Average mask probability per query, restricted to the
            # positive part of the mask logit (mirrors HF's behaviour).
            mask_sigmoid = msk.sigmoid()                    # (Q, h, w)
            mask_binary = (msk > 0).float()                  # (Q, h, w)
            mask_scores = (
                (mask_sigmoid.flatten(1) * mask_binary.flatten(1)).sum(1)
                / (mask_binary.flatten(1).sum(1) + 1e-6)
            )
            final_scores = scores_per_query * mask_scores    # (Q,)
            mask_h, mask_w = target_sizes[b]
            upsampled = torch.nn.functional.interpolate(
                mask_binary.unsqueeze(1), size=(mask_h, mask_w),
                mode="nearest",
            )[:, 0]                                        # (Q, H, W)

            bboxes_per_class = [[] for _ in range(self.num_classes)]
            masks_per_class = [[] for _ in range(self.num_classes)]

            for q in range(final_scores.shape[0]):
                score = float(final_scores[q].item())
                if score < score_threshold:
                    continue
                mask = upsampled[q].cpu().numpy().astype(np.uint8)
                if not mask.any():
                    continue
                bbox = _bbox_from_mask(mask.astype(bool))
                bbox = np.append(bbox, score).astype(np.float32)
                cid = int(labels_per_query[q].item())
                bboxes_per_class[cid].append(bbox)
                masks_per_class[cid].append(mask * 255)

            bboxes_arrays = []
            masks_arrays = []
            for cid in range(self.num_classes):
                if bboxes_per_class[cid]:
                    bboxes_arrays.append(
                        np.stack(bboxes_per_class[cid], axis=0).astype(np.float32)
                    )
                    masks_arrays.append(
                        np.stack(masks_per_class[cid], axis=0).astype(np.uint8)
                    )
                else:
                    bboxes_arrays.append(np.zeros((0, 5), dtype=np.float32))
                    masks_arrays.append(None)
            results.append((bboxes_arrays, masks_arrays))
        return results


def load_geneval(
    device: str = DEVICE,
    clip_arch: str = "ViT-L-14",
    clip_pretrained: str = "openai",
):
    """Construct a GenEval callable. Drop-in for the CUDA/mm-det original."""

    def timed(fn):
        def wrapper(*args, **kwargs):
            startt = time.time()
            result = fn(*args, **kwargs)
            endt = time.time()
            print(
                f"Function {fn.__name__!r} executed in {endt - startt:.3f}s",
                file=sys.stderr,
            )
            return result

        return wrapper

    @timed
    def load_models():
        seg = HFSegmenter(device=device)

        clip_model, _, transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained=clip_pretrained, device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_arch)

        with open(os.path.join(os.getcwd(), "reward_server/object_names.txt")) as cls_file:
            classnames = [line.strip() for line in cls_file]

        # Sanity check: the HF model's id2label order should match object_names.
        # We rely on this to keep bbox[i] aligned with classnames[i].
        return seg, (clip_model, transform, tokenizer), classnames

    COLORS = [
        "red", "orange", "yellow", "green", "blue",
        "purple", "pink", "brown", "black", "white",
    ]
    COLOR_CLASSIFIERS = {}

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            bgcolor = "#999"
            if bgcolor == "original":
                self._blank = self._image.copy()
            else:
                self._blank = Image.new("RGB", image.size, color=bgcolor)
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                assert (
                    tuple(self._image.size[::-1]) == tuple(mask.shape)
                ), (index, self._image.size[::-1], mask.shape)
                image = Image.composite(
                    self._image, self._blank, Image.fromarray(mask)
                )
            else:
                image = self._image
            image = image.crop(box[:4])
            return (transform(image), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model, tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                DEVICE,
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes),
            batch_size=16, num_workers=0,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(
            box[3] - box[1] + 1, 0
        )
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3]),
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised_offset = (
            np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0)
            * np.sign(offset)
        )
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5:  relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5:  relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, "
                            f"found {colors.count(req['color'])} {req['color']}; "
                            f"and " + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(
                            f"no target for {classname} to be {expected_rel}"
                        )
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, "
                                        f"found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, "
                    f"found {len(objects[classname])}"
                )
        return correct, "\n".join(reason)

    def evaluate_reward(image, objects, metadata):
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}=={req['count']}, "
                    f"found {len(found_objects)}"
                )
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(
                        1 - abs(req["count"] - colors.count(req["color"]))
                        / req["count"]
                    )
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, "
                            f"found {colors.count(req['color'])} {req['color']}; "
                            f"and " + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(
                            f"no target for {classname} to be {expected_rel}"
                        )
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, "
                                        f"found {' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    def evaluate_image(image_pils, metadatas, only_strict):
        results = object_detector.infer(
            image_pils, score_threshold=0.0, mask_threshold=0.5
        )
        ret = []
        for (bboxes, masks), image_pil, metadata in zip(results, image_pils, metadatas):
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            confidence_threshold = (
                THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            )
            for index, classname in enumerate(classnames):
                boxes_for_class = bboxes[index]
                if boxes_for_class.shape[0] == 0:
                    continue
                # Boxes arrive as [x1, y1, x2, y2, score]; sort desc by score.
                ordering = np.argsort(boxes_for_class[:, 4])[::-1]
                ordering = ordering[
                    boxes_for_class[ordering, 4] > confidence_threshold
                ]
                ordering = ordering[:MAX_OBJECTS].tolist()
                detected[classname] = []
                masks_for_class = masks[index] if masks[index] is not None else None
                while ordering:
                    max_obj = ordering.pop(0)
                    mask = (
                        None
                        if masks_for_class is None
                        else masks_for_class[max_obj].astype(bool)
                    )
                    detected[classname].append(
                        (boxes_for_class[max_obj].astype(np.float32), mask)
                    )
                    ordering = [
                        obj
                        for obj in ordering
                        if NMS_THRESHOLD == 1
                        or compute_iou(
                            boxes_for_class[max_obj], boxes_for_class[obj]
                        )
                        < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]

            is_strict_correct, score, reason = evaluate_reward(
                image, detected, metadata
            )
            if only_strict:
                is_correct = False
            else:
                is_correct, _ = evaluate(image, detected, metadata)
            ret.append({
                "tag": metadata["tag"],
                "prompt": metadata["prompt"],
                "correct": is_correct,
                "strict_correct": is_strict_correct,
                "score": score,
                "reason": reason,
                "metadata": json.dumps(metadata),
                "details": json.dumps({
                    key: [box.tolist() for box, _ in value]
                    for key, value in detected.items()
                }),
            })
        return ret

    object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def compute_geneval(images, metadatas, only_strict=False):
        required_keys = [
            "single_object", "two_object", "counting",
            "colors", "position", "color_attr",
        ]
        scores = []
        strict_rewards = []
        from collections import defaultdict
        grouped_strict_rewards = defaultdict(list)
        rewards = []
        grouped_rewards = defaultdict(list)
        results = evaluate_image(images, metadatas, only_strict=only_strict)
        for result in results:
            strict_rewards.append(1.0 if result["strict_correct"] else 0.0)
            scores.append(result["score"])
            rewards.append(1.0 if result["correct"] else 0.0)
            tag = result["tag"]
            for key in required_keys:
                if key != tag:
                    grouped_strict_rewards[key].append(-10.0)
                    grouped_rewards[key].append(-10.0)
                else:
                    grouped_strict_rewards[tag].append(
                        1.0 if result["strict_correct"] else 0.0
                    )
                    grouped_rewards[tag].append(
                        1.0 if result["correct"] else 0.0
                    )
        return scores, rewards, strict_rewards, dict(grouped_rewards), dict(
            grouped_strict_rewards
        )

    return compute_geneval


if __name__ == "__main__":
    # Minimal CLI: load model + run inference on a single image to confirm
    # the pipeline boots on the active device.
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="a photo of a person")
    parser.add_argument("--tag", default="single_object")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    fn = load_geneval(device=args.device)
    img = Image.open(args.image).convert("RGB")
    metadata = {
        "tag": args.tag,
        "prompt": args.prompt,
        "include": [{"class": args.prompt.split()[-1], "count": 1}],
    }
    scores, rewards, strict_rewards, group_rewards, group_strict_rewards = fn(
        [img], [metadata], only_strict=False
    )
    print(json.dumps({
        "scores": scores,
        "rewards": rewards,
        "strict_rewards": strict_rewards,
        "group_rewards": group_rewards,
        "group_strict_rewards": group_strict_rewards,
    }, indent=2))