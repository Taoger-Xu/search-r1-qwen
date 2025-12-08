# Enhanced Search-R1 Implementation: Improved Compatibility and Modern Framework Integration

## Overview

This repository provides an **enhanced and modernized implementation** of the [Search-R1 framework](https://github.com/PeterGriffinJin/Search-R1), originally developed for training **reasoning-and-searching interleaved LLMs**—language models that learn to reason and make tool calls (e.g., to search engines) in a coordinated manner.

The original Search-R1 framework, built upon [veRL](https://github.com/volcengine/verl), extends the ideas of **DeepSeek-R1(-Zero)** by incorporating interleaved search engine access and provides a fully open-source RL training pipeline. It serves as an alternative and open solution to **OpenAI DeepResearch**, enabling research and development in tool-augmented LLM reasoning.

Our enhanced implementation improves compatibility with modern frameworks and supports different RL methods (e.g., PPO, GRPO, reinforce), different LLMs (e.g., llama3, Qwen2.5, etc) and different search engines (e.g., local sparse/dense retrievers and online search engines).

Paper: [link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117); Model and data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full experiment log: [prelim](https://wandb.ai/peterjin/Search-R1-open); [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2); [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3).


## Our Contribution

This repository provides an **enhanced implementation** of the Search-R1 framework with the following key improvements:

### 🚀 **Modern Framework Integration**
- **Updated VERL Integration**: Full compatibility with latest vLLM (0.7.0+) and VERL frameworks, migrated from vLLM 0.6.3 to support modern environments
- **Advanced Model Support**: Native support for Qwen3-8B, Llama3.2, and DeepSeek models with enhanced GPU optimization
- **Enhanced LLM Generation Pipeline**: Comprehensive `LLMGenerationManager` class handling multi-turn conversations with integrated search capabilities
- **Multi-GPU Training**: Enhanced resource management with `_generate_with_gpu_padding()` for efficient multi-GPU scenarios

### 🔧 **Enhanced Search Integration**
- **Flexible Retrieval Architecture**: Multiple retriever implementations (Dense, BM25) with modular design for different search engines
- **Search API Integration**: Streamlined search API calls with batch processing optimization
- **Multi-Engine Support**: Local sparse/dense retrievers and online search engines with flexible configuration
- **Enhanced Search Pipeline**: Improved search result formatting and context integration


### 🛠️ **Enhanced Training and Evaluation Pipeline**
- **Streamlined Training Pipeline**: Enhanced model training, merging, and checkpoint management with VERL integration
- **Flexible Evaluation Scripts**: Training and evaluation scripts for both PPO and GRPO methods with comprehensive logging
- **Efficient Batch Evaluation**: Enabled large-scale inference and evaluation
- **Enhanced Metrics Collection**: Detailed performance tracking with W&B integration and structured logging
- **Production-Ready Pipeline**: Clean training scripts with automatic GPU detection and environment management


## Installation

Our Search-R1 implementation requires two separate environments: one for the main training framework (VERL) and another for the retrieval server. This separation ensures optimal performance and avoids dependency conflicts.

### Prerequisites

- CUDA-compatible GPU (minimum: 1 GPU with 8GB+ VRAM, recommended: multiple GPUs)
- CUDA 12.1 or higher
- Python 3.9-3.10
- Sufficient GPU memory for training (8GB+ per GPU recommended)

### 1. VERL Training Environment

This environment handles the main Search-R1 training pipeline with modern vLLM and VERL integration.

```bash
# Step 1: Create conda environment
conda create -n searchr1-verl python=3.9
conda activate searchr1-verl

# Step 2: Clone and setup VERL repository
git clone https://github.com/volcengine/verl.git
cd verl

# Step 3: Install vLLM, SGLang, and core dependencies
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Step 4: Install NVIDIA Apex for optimized training
git clone https://github.com/NVIDIA/apex.git
cd apex
CUDA_HOME=/usr/local/cuda MAX_JOB=64 pip install -v \
    --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./

# Step 5: Install VERL in development mode
cd ..
pip install --no-deps -e .

# Step 6: Install additional dependencies
pip install wandb  # for experiment tracking

# Step 7: Return to the Search-R1 repository
cd ..
```

### 2. Retrieval Server Environment

This environment runs the local dense retriever server for search functionality. We recommend using a separate environment to avoid conflicts.

```bash
# Step 1: Create retrieval environment
conda create -n searchr1-retriever python=3.10
conda activate searchr1-retriever

# Step 2: Install PyTorch with CUDA support (recommended via conda for FAISS compatibility)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# Step 3: Install core retrieval dependencies
pip install transformers datasets pyserini

# Step 4: Install FAISS-GPU for efficient vector search
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Step 5: Install API server dependencies
pip install uvicorn fastapi
```

### 3. Verification (Optional)

To verify your installation, run the following commands:

```bash
# Test VERL environment
conda activate searchr1-verl
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import verl; print('VERL successfully imported')"

# Test retrieval environment
conda activate searchr1-retriever
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

### Environment Management

- **Training**: Use the VERL environment for all training, evaluation, and inference tasks
- **Retrieval**: Use the retriever environment only for running the local retrieval server
- **Switching**: Always activate the appropriate environment before running scripts

## Quick Start

This guide walks you through training a Search-R1 model on the Natural Questions (NQ) dataset using E5 as the retriever and Wikipedia as the corpus. This demonstrates the core Search-R1 framework for question-answering tasks with search-augmented reasoning.

### Prerequisites

- Both conda environments installed (see [Installation](#installation))
- Access to compute resources with GPU support
- Internet connection for downloading datasets and indices

### Step 1: Download and Prepare Data

Download the pre-built E5 index and Wikipedia corpus for the NQ dataset:

```bash
# Set your desired save path
save_path=data/nq_search

# Download the indexing files and corpus
python scripts/download.py --save_path $save_path

# Combine index parts into single file
cat $save_path/part_* > $save_path/e5_Flat.index
rm $save_path/part_*

# Extract the Wikipedia corpus
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 2: Process the NQ Dataset

Prepare the Natural Questions dataset for Search-R1 training:

```bash
# Process NQ data into Search-R1 format
python scripts/data_process/nq_search.py
```

This script converts the NQ dataset into the required format with search-enabled prompts and ground truth answers.

### Step 3: Launch the Retrieval Server

Start the local dense retriever server using the E5 model and Wikipedia corpus. We recommend using tmux to run the server in the background:

```bash
# Create a new tmux session for the retrieval server
tmux new-session -d -s retrieval-server

# Activate the retrieval environment in tmux
tmux send-keys -t retrieval-server "conda activate searchr1-retriever" Enter

# Launch the retrieval server in tmux
tmux send-keys -t retrieval-server "python search/retrieval_server.py \
    --index_path data/nq_search/e5_Flat.index \
    --corpus_path data/nq_search/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu" Enter
```

The server will start and be available at `http://127.0.0.1:8000`. 

**Tmux Management Commands:**
```bash
# View tmux sessions
tmux list-sessions

# Attach to the retrieval server session to view logs
tmux attach-session -t retrieval-server

# Detach from tmux session (Ctrl+B, then D)
# Kill the retrieval server session when done
tmux kill-session -t retrieval-server
```

**Alternative: Direct Terminal Method**
If you prefer not to use tmux, you can run the server directly in a terminal:
```bash
# Activate the retrieval environment
conda activate searchr1-retriever

# Launch the retrieval server directly
python search/retrieval_server.py \
    --index_path data/nq_search/e5_Flat.index \
    --corpus_path data/nq_search/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```
Keep this terminal running during training.

**Monitor the retrieval server startup:**

```bash
# Check if server is running (in another terminal)
curl http://127.0.0.1:8000/health
```

**Test retrieval functionality:**

```bash
# Test Wikipedia search
curl -s -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["What is the capital of France?"],"topk":3,"return_scores":true}' | jq .
```

### Step 4: Run Training

**Important**: Training must run while the retrieval server is running since it connects to `localhost:8000`. Open another terminal and activate the VERL environment:

```bash
# Activate the VERL environment (in a new terminal)
conda activate searchr1-verl
```

### Step 5: Run Search-R1 Training

Choose your training method：


**Option A: PPO Training**
```bash
bash train_ppo.sh
```

**Option B: GRPO Training (Recommended)**
```bash
bash train_grpo.sh
```

**Monitor training progress:**

```bash
# Watch training logs (PPO)
tail -f logs/train_ppo.log

# Watch training logs (GRPO)
tail -f logs/train_grpo.log

# Monitor GPU usage
nvidia-smi

# Check Weights & Biases dashboard (if configured)
```

**Model Merge:**
After training, the output is a VERL FSDP checkpoint. The next step is to merge this into a HuggingFace format model for deployment and evaluation.
```bash
bash merge.sh
```

### Step 6: Evaluation and Analysis

After training completes, evaluate the trained model using the appropriate evaluation script:

```bash
# For PPO-trained models
bash eval_ppo.sh
# Output: inference/nq_search_results.jsonl

# For GRPO-trained models  
bash eval_grpo.sh
# Output: inference/nq_grpo_search_results.jsonl
```

**Evaluation Outputs:**
- **Performance Metrics**: Exact Match (EM) scores, search behavior statistics
- **Detailed Logs**: Training progress in `logs/eval_nq-search-r1-[method]_[timestamp].log`
- **Result Files**: JSONL files containing model predictions with search traces
- **W&B Dashboard**: Real-time metrics and visualizations (if configured)


**Common Issues:**
- **Connection refused**: Ensure retrieval server is running on port 8000
- **CUDA OOM**: Reduce batch size in training configuration
- **Missing files**: Re-run download script or check file paths

## Use your own dataset

### QA data
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

You can refer to ```scripts/data_process/nq_search.py``` for a concrete data processing example.

### Corpora

It is recommended to make your corpus a jsonl file, where each line (a dictionary with "id" key and "contents" key) corresponds to one passage. The downloaded Wikipedia corpus (`data/nq_search/wiki-18.jsonl`) serves as a reference example.

The "id" key corresponds to the passage id, while the "contents" key corresponds to the passage content ('"' + title + '"\n' + text).
For example:
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**Index your corpora (optional).**
If you would like to use a local retriever as the search engine, you can index your own corpus by:
```
bash search/build_index.sh
```
You can change ```retriever_name``` and ```retriever_model``` to your interested off-the-shelf retriever.

## Use your own search engine

Our codebase supports local sparse retriever (e.g., BM25), local dense retriever (both flat indexing with GPUs and ANN indexing with CPUs) and online search engine (e.g., Google, Bing, etc). More details can be found [here](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md).

The main philosophy is to launch a local or remote search engine server separately from the main RL training pipeline. 

The LLM can call the search engine by calling the search API (e.g., "http://127.0.0.1:8000/retrieve").

You can refer to ```search/retrieval_server.py``` for an example of launching a local retriever server.

## Features
- Support local sparse retrievers (e.g., BM25). ✔️
- Support local dense retrievers (both flat indexing and ANN indexing) ✔️
- Support google search / bing search / brave search API and others. ✔️
- Support off-the-shelf neural rerankers. ✔️
- Support different RL methods (e.g., PPO, GRPO, reinforce). ✔️
- Support different LLMs (e.g., llama3, Qwen2.5, etc). ✔️

## Contributors

- [Xinyi Zhao](https://github.com/Xinyi-0724) (Primary & Corresponding: xyzhao24@uw.edu)
- [Jinfeng Xiao](https://github.com/JinfengXiao)

