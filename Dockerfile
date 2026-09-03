# ChamNet — verified environment.
#
# mmcv has no prebuilt wheel for the torch version this project was validated
# against, so it is compiled from source here. That is the slow step (tens of
# minutes); everything after it is quick, and it is cached across rebuilds.
#
# Base: NVIDIA's PyTorch 25.09 container, which supplies python 3.12 and the
# torch 2.9 build this project's numbers were produced with.
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Which GPU architectures mmcv's CUDA kernels are compiled for. The default
# covers Turing through Blackwell; narrow it to your own card to cut build time
# roughly proportionally, e.g. --build-arg TORCH_CUDA_ARCH_LIST=8.6 for RTX 30xx.
#   7.5 = T4, RTX 20xx     8.0 = A100        8.6 = RTX 30xx, A40
#   8.9 = RTX 40xx, L40    9.0 = H100        12.0 = Blackwell, RTX PRO
# The trailing +PTX keeps the image usable on architectures newer than this list.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0+PTX"
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    MMCV_WITH_OPS=1 \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Pinned to the versions the reported results were produced with. pytest and
# Pillow are the [test] extra, declared here rather than relied on: chamnet is
# installed with --no-deps further down, so the extra is skipped, and `chamnet
# smoke` shells out to pytest while tests/conftest.py imports PIL. This base
# image happens to ship both, which is luck rather than a contract.
#
# opencv-python-headless is inert here and is kept only because the environment
# these results were produced in had it installed: the NGC base image ships its
# own OpenCV build earlier on sys.path, so `import cv2` resolves to that one
# (5.0.0) in this image and in the machine the numbers came from alike. Do not
# spend time reconciling this pin with what `cv2.__version__` reports.
RUN python -m pip install \
        mmengine==0.10.7 \
        timm==1.0.19 \
        pandas==2.2.3 \
        scipy==1.15.3 \
        prettytable==3.16.0 \
        opencv-python-headless==4.11.0.86 \
        albumentations==2.0.8 \
        pycocotools==2.0.10 \
        ftfy==6.3.1 regex==2024.11.6 \
        pytest Pillow

# The slow layer. Kept before the source copy so editing the package does not
# trigger a recompile.
RUN git clone --branch v2.1.0 --depth 1 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    python -m pip install -v /tmp/mmcv && \
    rm -rf /tmp/mmcv

# Upstream mmsegmentation, unmodified. This project is a package on top of it,
# not a fork of it.
RUN python -m pip install mmsegmentation==1.2.2

# Last two layers on purpose: they are the only ones a source edit
# invalidates, so editing the package rebuilds in about a second while the
# mmcv compile above stays cached. Anything added to the pinned-dependency
# layer near the top instead costs the full ~12-minute rebuild.
COPY . /opt/chamnet
# `pip install <directory>` builds in place, leaving build/ and
# chamnet.egg-info/ behind inside /opt/chamnet -- and build/lib/ holds copies
# of every module. They are install turds sitting next to the source a reader
# is invited to run `pytest` from, so they go in the same layer that creates
# them rather than becoming part of the image.
RUN python -m pip install --no-deps /opt/chamnet && \
    rm -rf /opt/chamnet/build /opt/chamnet/*.egg-info

WORKDIR /workspace
