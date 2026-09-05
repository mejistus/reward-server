from PIL import Image
from io import BytesIO
import pickle
import traceback
import os

# Switch backend to the MPS-friendly implementation when running on
# Apple Silicon (or anywhere mmdet/mmcv isn't available). Setting
# GENEVAL_BACKEND=mmdet keeps the legacy CUDA/mm-det path.
if os.environ.get("GENEVAL_BACKEND", "mps") == "mps":
    from reward_server.gen_eval_mps import load_geneval
else:
    from reward_server.gen_eval import load_geneval

import numpy as np

from flask import Flask, request, Blueprint

root = Blueprint("root", __name__)

def create_app():
    global INFERENCE_FN
    INFERENCE_FN = load_geneval()

    app = Flask(__name__)
    app.register_blueprint(root)
    return app

@root.route("/", methods=["POST"])
def inference():
    print(f"received POST request from {request.remote_addr}")
    data = request.get_data()

    try:
        data = pickle.loads(data)

        images = [Image.open(BytesIO(d), formats=["jpeg"]) for d in data["images"]]
        meta_datas = data["meta_datas"]
        only_strict = data["only_strict"]

        print(f"Got {len(images)} images")

        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = INFERENCE_FN(images, meta_datas, only_strict)

        response = {"scores": scores, "rewards": rewards, "strict_rewards": strict_rewards, "group_rewards": group_rewards, "group_strict_rewards": group_strict_rewards}

        response = pickle.dumps(response)

        returncode = 200
    except Exception as e:
        response = traceback.format_exc()
        print(response)
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


HOST = "127.0.0.1"
PORT = 8085

if __name__ == "__main__":
    create_app().run(HOST, PORT)