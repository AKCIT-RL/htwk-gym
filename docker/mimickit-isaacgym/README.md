# MimicKit + Isaac Gym Preview 4

Imagem mínima para validar o backend Isaac Gym do MimicKit. O pacote licenciado do Isaac Gym não é versionado nem incluído na imagem-fonte.

## Pré-requisito

Coloque `IsaacGym_Preview_4_Package.tar.gz` na raiz do workspace. O build usa a raiz como contexto:

```bash
docker build \
  -f docker/mimickit-isaacgym/Dockerfile \
  -t mimickit-isaacgym:p4 .
```

A imagem fixa Python 3.8, PyTorch CUDA 11.8 e dependências mínimas do AMP. `diffusers` é omitido porque não participa do AMP clássico.

## Smoke headless do aceno

```bash
docker run --rm --gpus all \
  --shm-size=1g \
  mimickit-isaacgym:p4 \
  python3.8 mimickit/run.py \
    --arg_file args/view_motion_g1_wave_args.txt \
    --num_envs 1 \
    --visualize false \
    --devices cuda:0 \
    --test_episodes 1
```

## Viewer X11

Autorize temporariamente o root local no X server:

```bash
xhost +SI:localuser:root
```

Depois execute:

```bash
docker run --rm --gpus all \
  --shm-size=1g \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  mimickit-isaacgym:p4 \
  python3.8 mimickit/run.py \
    --arg_file args/view_motion_g1_wave_args.txt \
    --num_envs 1 \
    --visualize true \
    --devices cuda:0 \
    --test_episodes 1
```

Revogue a autorização ao terminar:

```bash
xhost -SI:localuser:root
```

Não use `--privileged`. O smoke permanece em um ambiente para limitar consumo de VRAM.
