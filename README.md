# Nemotron 3 Super：BF16 → FP8 PTQ（Singularity / Apptainer + Slurm）

使用 **官方 Megatron-Bridge（`super-v3`）+ Model Optimizer PTQ**，在 HPC 上把
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` 量化為本機 FP8 Hugging Face export。
**不是**單純 dtype cast。

> **EN (brief):** Singularity/Apptainer + Slurm workflow for official Megatron-Bridge
> Model Optimizer FP8 PTQ (`mamba_moe_fp8_conservative` / `_aggressive`). Prefer NGC image
> `nvcr.io/nvidia/nemo:26.02.nemotron_3_super`. See [docs/hpc-notes.md](docs/hpc-notes.md)
> and NVIDIA docs linked below. Do not commit weights or `.sif` files.

## 官方流程摘要

1. **映像**：優先 `nvcr.io/nvidia/nemo:26.02.nemotron_3_super`（`singularity/apptainer pull`）。
   Megatron-Bridge 根目錄通常為 `/opt/Megatron-Bridge`；量化需 **`super-v3`**。
2. **量化設定**：`mamba_moe_fp8_conservative`（預設）與 `mamba_moe_fp8_aggressive`。
3. **指令模式**（與上游一致）：

```bash
torchrun --nproc_per_node=${GPUS_PER_NODE} examples/quantization/quantize.py \
  --hf-model-id $HF_MODEL \
  --export-quant-cfg mamba_moe_fp8_conservative \
  --megatron-save-path $MEGATRON_SAVE_PATH \
  --pp $PP --tp $TP --ep $EP --trust-remote-code
```

可選驗證：`examples/quantization/ptq_generate.py`。

匯出 HF：

```bash
torchrun --nproc_per_node=${GPUS_PER_NODE} examples/quantization/export.py \
  --hf-model-id $HF_MODEL \
  --megatron-load-path $MEGATRON_SAVE_PATH \
  --export-dir $EXPORT_DIR \
  --pp $EXPORT_PP --dtype bfloat16 --trust-remote-code
```

文件說明：FP8 calibration 約 **256** 筆 SFT samples；實務上常需至少 **2×8 H100**；
TP / PP / EP 可調。官方亦已在 HF 釋出 FP8 權重；本 repo 著重 **自行 PTQ 匯出**。

## 目錄結構

```
configs/fp8_conservative.env   # 預設 FP8 conservative
configs/fp8_aggressive.env
singularity/                   # .def + pull 說明
scripts/                       # pull / prepare / quantize / generate / export
slurm/                         # sbatch + 相依 pipeline
docs/hpc-notes.md
.env.example
```

## Quickstart

```bash
git clone https://github.com/alex-ht/nemotron3-super-bf16-fp8-hpc.git
cd nemotron3-super-bf16-fp8-hpc

cp .env.example .env
# 編輯 .env：路徑、SLURM_PARTITION / ACCOUNT、SIF_PATH、HF_HOME 等

export CONFIG=$PWD/configs/fp8_conservative.env

./scripts/00_pull_or_build_sif.sh          # 需要 NGC 憑證
./scripts/01_prepare_dirs.sh

# 建議：掛載本機 super-v3
# git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git && cd Megatron-Bridge && git checkout super-v3
# export MEGATRON_BRIDGE_HOST=$PWD

# 下載或放置 BF16 模型到 MODEL_DIR / HF_HOME 後：
./slurm/submit_pipeline.sh                 # quantize → generate → export
# 或略過 generate：
# SKIP_GENERATE=1 ./slurm/submit_pipeline.sh
```

單步 sbatch：

```bash
sbatch --export=ALL,CONFIG=$PWD/configs/fp8_conservative.env,REPO_ROOT=$PWD \
  slurm/quantize.sbatch
sbatch --dependency=afterok:<QJOB> --export=ALL,CONFIG=...,REPO_ROOT=$PWD \
  slurm/export.sbatch
```

日誌寫入 `logs/`。

## 平行化建議表

| 情境 | NNODES | GPUS_PER_NODE | TP | PP | EP | 備註 |
|------|--------|---------------|----|----|----|------|
| 文件常見最小值 | 2 | 8 | 8 | 2 | 8 | 2×8 H100 |
| 單節點（較緊） | 1 | 8 | 8 | 1 | 8 | 視顯存與實作而定 |
| Export（文件範例） | ≥1 | 8 | 1* | 8* | 1* | 使用 `EXPORT_*`；以官方 `export.py` 為準 |

\* Export 的 `--tp/--pp/--ep` 由 `EXPORT_TP` / `EXPORT_PP` / `EXPORT_EP` 控制，可與 quantize 不同。

## 疑難排解（Troubleshooting）

| 症狀 | 可能原因 | 處理 |
|------|----------|------|
| 找不到 SIF | 尚未 pull | `./scripts/00_pull_or_build_sif.sh` + NGC token |
| `quantize.py` 不存在 / 舊旗標 | 映像內非 super-v3 | 設定 `MEGATRON_BRIDGE_HOST` 掛載 host checkout |
| NCCL hang | 跨節點網路 / 容器未 `--nv` | 檢查 IB、`--nv`、各節點皆能讀 SIF |
| OOM | 平行度不足或 PP 太小 | 增加節點或提高 `PP` |
| HF 下載失敗 | 未授權 | 設定 `HF_TOKEN`；確認模型授權 |
| Slurm 資源不符 | `#SBATCH` 與 env 不一致 | 用 `submit_pipeline.sh` 覆蓋 `--nodes` / `--gpus-per-node` |

## 授權（License）

- 本倉庫腳本與文件：**Apache-2.0**（見 [LICENSE](LICENSE)）。
- **NVIDIA 模型權重、NGC 容器、NeMo / Megatron-Bridge 等** 受 **另行 NVIDIA 授權 / EULA** 約束；使用前請自行取得並遵守。

## 參考連結

- https://docs.nvidia.com/nemo/megatron-bridge/latest/models/nemotron/nemotron3-super.html
- https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html
- https://github.com/NVIDIA-NeMo/Megatron-Bridge (branch `super-v3`)
- HF BF16：`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
