# Singularity / Apptainer for Nemotron 3 Super FP8 PTQ

## Preferred: pull NGC image

NVIDIA documents the custom container:

`nvcr.io/nvidia/nemo:26.02.nemotron_3_super`

```bash
# Authenticate to NGC once (API key from https://ngc.nvidia.com)
# export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
# export SINGULARITY_DOCKER_PASSWORD=<NGC_API_KEY>

export SIF_PATH="${SIF_PATH:-$PWD/../scratch/nemo_26.02.nemotron_3_super.sif}"
mkdir -p "$(dirname "$SIF_PATH")"

# Apptainer (or singularity)
apptainer pull "$SIF_PATH" docker://nvcr.io/nvidia/nemo:26.02.nemotron_3_super
# singularity pull "$SIF_PATH" docker://nvcr.io/nvidia/nemo:26.02.nemotron_3_super
```

Or use the helper:

```bash
./scripts/00_pull_or_build_sif.sh
```

## Optional: build from `.def`

```bash
apptainer build --nv "$SIF_PATH" singularity/nemotron3-super.def
```

Building from DockerHub/NGC still requires NGC credentials and is slower than `pull`.

## Megatron-Bridge `super-v3`

Quantization support requires the latest `super-v3` branch of
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).

Inside the container the tree is often at `/opt/Megatron-Bridge`. Options:

1. **Mount host checkout** (recommended for reproducibility):

   ```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
   cd Megatron-Bridge && git checkout super-v3
   export MEGATRON_BRIDGE_HOST=$PWD
   # scripts bind-mount this to /opt/Megatron-Bridge when set
   ```

2. **Update in-place** inside an interactive shell (if the image path is a git repo):

   ```bash
   cd /opt/Megatron-Bridge   # or /opt/megatron per image notes
   git pull origin super-v3
   ```

## GPU + bind mounts

Always use `--nv` (or equivalent) and bind model / ckpt / HF cache / workspace:

```bash
apptainer exec --nv \
  --bind "$MODEL_DIR:$MODEL_DIR" \
  --bind "$MEGATRON_SAVE_PATH:$MEGATRON_SAVE_PATH" \
  --bind "$HF_HOME:$HF_HOME" \
  --bind "$WORKSPACE:$WORKSPACE" \
  --bind "$REPO_ROOT:$REPO_ROOT" \
  ${MEGATRON_BRIDGE_HOST:+--bind "$MEGATRON_BRIDGE_HOST:/opt/Megatron-Bridge"} \
  "$SIF_PATH" bash
```

Working directory for PTQ scripts: `/opt/Megatron-Bridge`.

## References

- [Megatron-Bridge Nemotron 3 Super](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/nemotron/nemotron3-super.html)
- [Nemotron Stage 3 Quantization](https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html)
- NGC catalog: search `nemo` / `nemotron_3_super`
