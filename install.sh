#!/usr/bin/env bash
uv pip install torch==2.1.2 torchvision==0.16.2
uv pip install "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html --no-build-isolation
uv pip install "mmdet==2.28.2"


git clone https://github.com/open-mmlab/mmdetection.git mmdetection2
cd mmdetection2
git checkout v2.28.2
