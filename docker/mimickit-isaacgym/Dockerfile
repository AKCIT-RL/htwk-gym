FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ARG UV_VERSION=0.8.15
ARG TORCH_VERSION=2.4.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONPATH=/opt/isaacgym/python:/workspace/MimicKit/mimickit \
    LD_LIBRARY_PATH=/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        libgl1 \
        libglu1-mesa \
        libpython3.8 \
        libsm6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

COPY IsaacGym_Preview_4_Package.tar.gz /tmp/isaacgym.tar.gz
RUN mkdir -p /opt/isaacgym \
    && tar -xzf /tmp/isaacgym.tar.gz -C /opt/isaacgym --strip-components=1 \
    && rm /tmp/isaacgym.tar.gz

COPY docker/mimickit-isaacgym/requirements-amp-py38.txt /tmp/requirements-amp.txt
RUN uv pip install --system --python /usr/bin/python3.8 \
        --index-url https://download.pytorch.org/whl/cu118 \
        "torch==${TORCH_VERSION}+cu118" "torchvision==0.19.1+cu118" \
    && uv pip install --system --python /usr/bin/python3.8 -r /tmp/requirements-amp.txt \
    && uv pip install --system --python /usr/bin/python3.8 --no-deps /opt/isaacgym/python \
    && uv pip check --python /usr/bin/python3.8

WORKDIR /workspace/MimicKit
COPY MimicKit/ /workspace/MimicKit/

RUN python3.8 -c "from isaacgym import gymapi, gymtorch; import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('isaacgym imports OK')" \
    && PYTHONPATH=/workspace/MimicKit/mimickit python3.8 -c "from anim.motion import Motion; print('MimicKit imports OK')"

CMD ["/bin/bash"]
