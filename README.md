# search-r1-qwen

[English](README_EN.md) | 中文

基于 veRL 的 retrieval-augmented RL 项目，支持 PPO 和 GRPO，让模型交替执行 reasoning、search 和 answer generation。默认使用 Qwen2.5-3B Base、E5 retriever 与 Wikipedia corpus。

## Installation

环境要求：Linux、NVIDIA GPU、CUDA 12、Python 3.10。训练与检索环境分别安装，路径按需修改。

### Search-r1 environment

```bash
cd /path/to/search-r1-qwen
uv venv --python 3.10 .venv-train
uv pip install --python .venv-train/bin/python \
  -r requirements-train.txt
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

使用仓库自带的 `verl/`。FlashAttention 的 prebuilt wheel 适用于 Linux x86_64、Python 3.10 和 PyTorch 2.6。

### Retrieval Server Environment

安装 retrieval server 的依赖：

```bash
uv venv --python 3.10 .venv-retriever
uv pip install --python .venv-retriever/bin/python \
  -r requirements-retriever.txt
```

## 数据准备

下载 model weights、NQ dataset、Wikipedia corpus 和 E5 index，再将 dataset 转换为 training 所需的 Parquet 格式：

```bash
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/download_local_assets.py
python scripts/prepare_local_data.py
python scripts/data_process/nq_search.py \
  --dataset_dir /path/to/datasets/nq \
  --local_dir /path/to/datasets/nq_search
```

model weights 和 dataset 的保存位置按需配置，完整 FAISS index 约 64.6 GB。
论文实验的 dataset 与 validation 配置见 [实验说明](docs/paper-reproduction.md)。

## Retrieval Server

在独立终端启动 retrieval server，等待 FAISS index 加载完成：

```bash
bash scripts/retrieval/start_local.sh
curl --fail http://127.0.0.1:8000/health
```

默认使用 CPU FAISS。先将 `RETRIEVER_GPU` 设置为 retrieval 使用的 GPU ID，再启用 GPU FAISS：

```bash
RETRIEVER_GPU="$RETRIEVER_GPU" RETRIEVER_FAISS_GPU=1 bash scripts/retrieval/start_local.sh
```

## 训练与评估

先启动 retrieval server，再选择 PPO 或 GRPO。将 `TRAIN_GPUS` 设置为 training 使用的 GPU IDs（以逗号分隔）；启用 GPU FAISS 时，training 与 retrieval 应使用不同 GPU。

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/ppo/train.sh
# 或
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/grpo/train.sh
```

training 完成后执行 checkpoint merge 和 evaluation（PPO 将 `grpo` 改为 `ppo`）：

```bash
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/common/merge.sh grpo
bash scripts/grpo/eval.sh
```

merge 脚本默认使用 step 100 的 checkpoint。论文实验入口位于 `scripts/experiments/`，完整命令见 [实验说明](docs/paper-reproduction.md)。
日志统一保存在 `logs/`，不纳入 Git；checkpoint 保存在 `verl_checkpoints/`。

## Contributors

- xujintao

## 致谢与许可

基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)、[Search-R1-Qwen3](https://github.com/Xinyi-0724/Search-R1-Qwen3) 和 veRL。
许可及版权声明见 [LICENSE](LICENSE) 和 [Notice.txt](Notice.txt)。
