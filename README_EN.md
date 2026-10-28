# search-r1-qwen

English | [中文](README.md)

Retrieval-augmented reinforcement learning with veRL, supporting PPO and GRPO. The default setup uses Qwen2.5-3B Base, an E5 retriever, and Wikipedia.

## Installation

Requirements: Linux, NVIDIA GPU, CUDA 12, and Python 3.10. Use separate training and retrieval environments; adjust paths as needed.

### Search-r1 environment

```bash
cd /path/to/search-r1-qwen
uv venv --python 3.10 .venv-train
uv pip install --python .venv-train/bin/python \
  -r requirements-train.txt
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Use the bundled `verl/`. The pinned FlashAttention wheel targets Linux x86_64, Python 3.10, and PyTorch 2.6.

### Retrieval Server Environment

Install retrieval dependencies:

```bash
uv venv --python 3.10 .venv-retriever
uv pip install --python .venv-retriever/bin/python \
  -r requirements-retriever.txt
```

## Data Preparation

Download the models, NQ dataset, Wikipedia corpus, and E5 index, then prepare the training data:

```bash
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/download_local_assets.py
python scripts/prepare_local_data.py
python scripts/data_process/nq_search.py \
  --dataset_dir /path/to/datasets/nq \
  --local_dir /path/to/datasets/nq_search
```

Choose model and dataset storage paths for your environment. The full retrieval index is approximately 64.6 GB.
See [experiment notes](docs/paper-reproduction.md) for the paper dataset and validation setup.

## Retrieval Server

Start the server in a separate terminal and wait for the index to load:

```bash
bash scripts/retrieval/start_local.sh
curl --fail http://127.0.0.1:8000/health
```

CPU FAISS is the default. Set `RETRIEVER_GPU` to your retrieval device ID before enabling GPU FAISS:

```bash
RETRIEVER_GPU="$RETRIEVER_GPU" RETRIEVER_FAISS_GPU=1 bash scripts/retrieval/start_local.sh
```

## Training and Evaluation

Start retrieval first, then choose PPO or GRPO. Set `TRAIN_GPUS` to a comma-separated list of training device IDs; exclude the retrieval GPU when GPU FAISS is enabled.

```bash
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/ppo/train.sh
# Or
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" bash scripts/grpo/train.sh
```

Merge the checkpoint and evaluate after training. Replace `grpo` with `ppo` for PPO:

```bash
source .venv-train/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/common/merge.sh grpo
bash scripts/grpo/eval.sh
```

The merger defaults to the step-100 checkpoint. Paper experiment launchers are under `scripts/experiments/`; see [experiment notes](docs/paper-reproduction.md) for full commands.
Logs are stored in `logs/` and excluded from Git. Checkpoints are stored in `verl_checkpoints/`.

## Contributors

- xujintao

## Acknowledgments and License

Based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [Search-R1-Qwen3](https://github.com/Xinyi-0724/Search-R1-Qwen3), and veRL.
See [LICENSE](LICENSE) and [Notice.txt](Notice.txt) for licensing and copyright notices.
