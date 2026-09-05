# reward-server

HTTP server that serves GenEval and DeQA reward inference. GenEval supports two
backends:

* **MPS** (default) — pure PyTorch + 🤗 Transformers, runs on Apple Silicon via
  MPS and on any GPU/CPU as a fallback. **No `mmdet` / `mmcv` / `mmengine`**.
* **mmdet** (legacy) — the original CUDA-only path that uses the
  mmdetection Mask2Former checkpoint.

`GENEVAL_BACKEND=mps` is the default in both `app_geneval.py` and
`gunicorn.conf.py`; flip to `mmdet` to restore the legacy path.

---

## Quick start — one-stop setup

### GenEval on Apple Silicon (MPS)

```bash
git clone https://github.com/mejistus/reward-server.git
cd reward-server

# 1. Fresh env
python -m venv .venv && source .venv/bin/activate

# 2. PyTorch (Apple Silicon build)
pip install --upgrade torch torchvision

# 3. Reward stack (no mmdet / mmcv / mmengine)
pip install --upgrade transformers open_clip_torch clip_benchmark \
                    flask gunicorn pillow numpy

# 4. Pre-warm the HF Mask2Former + OpenAI CLIP weights so the first request
#    isn't slowed down by downloads (the Mask2Former blob is ~275 MB, the
#    OpenAI CLIP-L is ~890 MB; tune HF_HOME / XDG_CACHE_HOME to relocate).
HF_HOME=~/.cache/huggingface python - <<'PY'
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance"
)
Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-coco-instance"
)
PY
mkdir -p ~/.cache/clip
curl -sSL -o ~/.cache/clip/ViT-L-14.openai.pt \
    https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

# 5. Smoke test over a folder of images
python test/test_geneval_mps.py --image-dir ~/Pictures/diffusion

# 6. Start the server (single MPS worker; edit gunicorn.conf.py to bind 0.0.0.0
#    if you need cross-node access from training)
gunicorn "app_geneval:create_app()"
```

### GenEval on CUDA (mmdet, legacy)

```bash
git clone https://github.com/mejistus/reward-server.git
cd reward-server

# 1. Conda env (Python 3.10.16 is what the original repo used)
conda create -n reward_server python=3.10.16 -y
conda activate reward_server

# 2. PyTorch + CUDA 12.1
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
            --index-url https://download.pytorch.org/whl/cu121

# 3. Reward stack
pip install gunicorn==23.0.0 openmim==0.3.9 open-clip-torch==2.31.0 \
            numpy==1.26.0 opencv-python==4.11.0.86 \
            clip-benchmark==1.6.1 flask==3.1.0

# 4. mmdet / mmcv (legacy CUDA path)
mim install mmcv-full mmengine
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && git checkout 2.x
# Edit mmdet/__init__.py: set mmcv_maximum_version = '2.3.0'
pip install -e .
cd ../reward-server

# 5. Download Mask2Former weights
mkdir -p ./model/mask2former2
wget -O ./model/mask2former2/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth \
    https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth

# 6. Pick the legacy backend and start the server
export GENEVAL_BACKEND=mmdet
# Edit MY_CONFIG_PATH / MY_CKPT_PATH in reward_server/gen_eval.py if needed,
# and set NUM_DEVICES (env var) in gunicorn.conf.py for multi-GPU.
gunicorn "app_geneval:create_app()"
```

For H-type GPUs you may hit `error in ms_deformable_im2col_cuda: no kernel
image is available for execution on the device`. Fix:

```bash
mim uninstall mmcv-full
TORCH_CUDA_ARCH_LIST="9.0" pip install mmcv-full
```

### DeQA (CUDA)

```bash
# DeQA has its own deps; see https://github.com/zhiyuanyou/DeQA-Score
git clone https://github.com/mejistus/reward-server.git
cd reward-server
conda activate reward_server            # reuse the env above, or create your own
# Install DeQA's own requirements, then:
gunicorn "app_deqa:create_app()"
```

---

## Environment variables

| Var                  | Default                                          | Notes                                            |
|----------------------|--------------------------------------------------|--------------------------------------------------|
| `GENEVAL_BACKEND`    | `mps`                                            | Set to `mmdet` to restore the legacy CUDA path.  |
| `GENEVAL_M2F_REPO`   | `facebook/mask2former-swin-small-coco-instance`  | Swap to another HF Mask2Former if you need to.   |
| `HF_HOME`            | `~/.cache/huggingface`                           | Where HF downloads the Mask2Former weights.      |
| `NUM_DEVICES`        | (only used by `mmdet` backend)                   | Set to number of GPUs you want gunicorn to spawn. |

---

## Usage

### GenEval

After `gunicorn "app_geneval:create_app()"` is running, sanity-check it:

```bash
# Original client (works against either backend)
python test/test_geneval.py

# New folder scanner for the MPS backend — no ground-truth prompts needed;
# metadata is bootstrapped from Mask2Former detections on the fly.
python test/test_geneval_mps.py --image-dir ~/Pictures/diffusion
```

Expected output format:

```python
{
  'scores':           [0.75, 1.0],
  'rewards':          [0.0, 1.0],
  'strict_rewards':   [0.0, 1.0],
  'group_rewards': {
    'single_object': [-10.0, 1.0], 'two_object': [-10.0, -10.0],
    'counting':      [-10.0, -10.0], 'colors':    [-10.0, -10.0],
    'position':      [-10.0, -10.0], 'color_attr':[0.0,    -10.0],
  },
  'group_strict_rewards': { ... },
}
```

Sample folder-scanner summary over `~/Pictures/diffusion`:

```
=== summary ===
  color_attr    n= 32  avg_reward=1.000  avg_strict=0.375  avg_score=-0.547
  counting      n= 29  avg_reward=0.000  avg_strict=0.000  avg_score=0.045
  single_object n= 35  avg_reward=1.000  avg_strict=0.486  avg_score=-1.914

processed=35  empty=0  images_dir=/Users/eculid/Pictures/diffusion  device=mps
```

### DeQA

After `gunicorn "app_deqa:create_app()"` is running, sanity-check it:

```bash
python test/test_deqa.py
```