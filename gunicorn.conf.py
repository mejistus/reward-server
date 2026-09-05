import os
import sys

# Backend selector: "mps" (default) routes GenEval through the HF/MPS
# implementation in reward_server/gen_eval_mps.py, "mmdet" keeps the
# legacy CUDA/mmdet path in reward_server/gen_eval.py.
reward = os.environ.get("GENEVAL_BACKEND", "mps")
if reward not in {"mps", "mmdet"}:
    raise SystemExit(f"GENEVAL_BACKEND must be one of mps|mmdet, got {reward!r}")
os.environ["GENEVAL_BACKEND"] = reward

if reward == "mps":
    # MPS has a single shared GPU device; multiple gunicorn workers
    # would all contend for it. Stay with one worker.
    NUM_DEVICES = 1
    port = 18085
else:
    NUM_DEVICES = int(os.environ.get("NUM_DEVICES", "8"))
    port = 18085

USED_DEVICES = set()


def pre_fork(server, worker):
    global USED_DEVICES
    worker.device_id = next(
        i for i in range(NUM_DEVICES) if i not in USED_DEVICES
    )
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    if reward == "mmdet":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)
    # For the MPS backend there is only one Apple GPU; we don't try to
    # pin a per-worker device.


def child_exit(server, worker):
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)


bind = f"127.0.0.1:{port}"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 300