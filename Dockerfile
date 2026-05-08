# --- Estágio 1: Imagem Base e Dependências Essenciais ---
# Imagem oficial NVIDIA CUDA 11.8 + CUDNN 8 (devel inclui headers/compiladores).
# Para rodar APENAS em CPU (sanity tests, CI), basta não passar --gpus.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

LABEL maintainer="htwk-gym" \
      description="Isaac Gym RL environment – T1 Kicking with running approach" \
      cuda="11.8" \
      python="3.8"

# Evita prompts interativos durante o build
ENV DEBIAN_FRONTEND=noninteractive

# Habilita capabilities gráficas da NVIDIA (necessário para renderização offscreen com Vulkan)
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

# Saída do Python não bufferizada (logs em tempo real no docker logs)
ENV PYTHONUNBUFFERED=1
# Evita geração de .pyc dentro do container
ENV PYTHONDONTWRITEBYTECODE=1

# Ubuntu 20.04 já inclui Python 3.8 nativamente — sem necessidade de PPA externo.
# Isso evita dependência de rede para chave GPG durante o build.
RUN apt-get update && \
    apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3-pip \
        git \
        vim \
        curl \
        htop \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        libvulkan1 \
        patchelf && \
    rm -rf /var/lib/apt/lists/*

# Define python3.8 como padrão (já é o default no ubuntu20.04, mas explicitamos)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --set python3 /usr/bin/python3.8

RUN python3 -m pip install --no-cache-dir --upgrade pip

# --- Estágio 2: Ambiente Python e Bibliotecas de RL ---
WORKDIR /app

# LD_LIBRARY_PATH de runtime: apenas o diretório de config do Python 3.8.
# NÃO incluir /usr/local/cuda/lib64/stubs aqui — em runtime com GPU real,
# o NVIDIA Container Toolkit injeta a libcuda.so.1 do driver do host.
# Stubs só são usados durante o build (pip install do isaacgym).
# NOTA: ENV fica APÓS pip install torch para não quebrar o cache do Docker.

# PyTorch 2.0 para CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118 \
    numpy==1.24.1

ENV LD_LIBRARY_PATH=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu

# --- Estágio 3: Instalação do Isaac Gym ---
ARG ISAACGYM_FILE=IsaacGym_Preview_4_Package.tar.gz
COPY ${ISAACGYM_FILE} /tmp/

RUN mkdir -p /opt/isaacgym && \
    tar -xf /tmp/${ISAACGYM_FILE} -C /opt && \
    rm /tmp/${ISAACGYM_FILE}

# Corrige incompatibilidade np.float → float (numpy >= 1.24)
RUN sed -i 's/np\.float\b/float/g' /opt/isaacgym/python/isaacgym/torch_utils.py

# Instala o isaacgym usando o stub temporariamente no LD_LIBRARY_PATH (build-time only).
# O symlink libcuda.so.1 → stub satisfaz o linker durante a compilação das extensões C++.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH \
    pip install --no-cache-dir /opt/isaacgym/python

# PYTHONPATH inicializado com valor fixo (não pode referenciar a si mesmo no ENV do Dockerfile)
ENV PYTHONPATH="/opt/isaacgym/python"

# --- Estágio 4: Instalação da Aplicação ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Logs are persisted via bind mount: -v $(pwd)/logs:/app/logs

# --- Estágio 5: Comando de Execução ---
# Padrão: bash interativo. Para treinar, passe o comando explicitamente.
# Exemplo GPU:   docker run --gpus all -v $(pwd)/logs:/app/logs htwk-gym python3 train.py --task T1/Kicking
# Exemplo CPU:   docker run -v $(pwd)/logs:/app/logs htwk-gym python3 train.py --task T1/Kicking --sim_device cpu --rl_device cpu
CMD ["bash"]