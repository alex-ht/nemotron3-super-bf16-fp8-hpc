# HPC notes — Nemotron 3 Super BF16 → FP8 PTQ

## What this is (and is not)

This workflow runs **official Model Optimizer post-training quantization (PTQ)** through
**Megatron-Bridge** (`super-v3`) scripts under `examples/quantization/`.

It is **not** a naive `dtype` cast (e.g. casting weights to FP8 without calibration).

References:

- [Megatron-Bridge — Nemotron 3 Super](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/nemotron/nemotron3-super.html)
- [Nemotron Stage 3: Quantization](https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html)
- [Megatron-Bridge `super-v3` docs source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/super-v3/docs/models/llm/nemotron3-super.md)

## Container

Prefer the NGC image:

`nvcr.io/nvidia/nemo:26.02.nemotron_3_super`

Pull with Apptainer/Singularity (`scripts/00_pull_or_build_sif.sh`). Always run with
`--nv` and bind mounts for model / Megatron ckpt / `HF_HOME` / workspace.

Megatron-Bridge root inside the image is typically `/opt/Megatron-Bridge`.
Quantization needs the **`super-v3`** branch — mount a host checkout via
`MEGATRON_BRIDGE_HOST` if the image tree is stale.

## Parallelism

Official docs note a practical minimum around **2×8 H100** for this model’s PTQ.
`TP` / `PP` / `EP` are configurable. For multi-node, increase `--pp` (e.g. `--pp 2`
on 2 nodes × 8 GPUs) and launch with Slurm.

| Stage | Typical flags | Notes |
|-------|---------------|-------|
| Quantize | `--tp 8 --pp 2 --ep 8` | Match total world size to GPUs |
| PTQ generate | same as quantize | Smoke test only |
| Export HF | often `--pp 8` (see docs) | Uses `EXPORT_TP/PP/EP` |

Configs live in `configs/fp8_conservative.env` and `configs/fp8_aggressive.env`
(`mamba_moe_fp8_conservative` / `mamba_moe_fp8_aggressive`).

## Calibration

NVIDIA’s FP8 checkpoint description: calibration used **~256 samples** from the
post-training SFT dataset. Scripts accept `--calib-size` (env `CALIB_SIZE`, default 256 here).

## Multi-node torchrun under Slurm

1. `MASTER_ADDR` = first host from `scontrol show hostnames "$SLURM_NODELIST"`.
2. One `srun` task per node; inside the container `torchrun` uses
   `--nnodes`, `--nproc_per_node`, `--node_rank=$SLURM_NODEID`,
   `--master_addr`, `--master_port`.
3. Parameterize `NNODES`, `GPUS_PER_NODE`, `TP`, `PP`, `EP`, partition, account via
   env files / `.env` / `sbatch --export`.

## Hugging Face models

- Input BF16: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- Official FP8 also exists on HF if you only need an already-quantized checkpoint
  for inference (this repo’s purpose is **local** BF16→FP8 PTQ export).

## Site tips

- Put large SIF / weights / HF cache on scratch or parallel FS; never commit them.
- Set `HF_TOKEN` for gated downloads; prefer cluster secret stores over committing tokens.
- If NCCL hangs: check IB/RoCE modules, `NCCL_DEBUG=INFO`, firewall between nodes,
  and that the SIF is readable on every node (shared FS or identical local copies).
- If OOM: lower micro-batch/calib concurrency (if exposed), raise PP, or add nodes.
- Keep `#SBATCH` resource lines aligned with `NNODES` / `GPUS_PER_NODE` in the env file
  (or rely on `submit_pipeline.sh` overrides).

## License reminder

Scripts in this repo: Apache-2.0. NVIDIA models, NGC containers, and NVIDIA software
use **separate** NVIDIA licenses / EULAs.
