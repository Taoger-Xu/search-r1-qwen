# Search-R1 如何基于 veRL 完成强化学习训练

本文以仓库当前的 NQ + Qwen2.5-3B Base + 本地检索配置为准。虽然仓库名包含 Qwen3，根目录训练脚本当前使用的是 Qwen2.5-3B。下文先说明训练数据，再说明奖励、算法和 veRL 中每个训练阶段的实际调用路径。

## 1. RL 数据的格式

### 1.1 磁盘上的一条样本

`scripts/data_process/nq_search.py` 从 NQ 的 `question`、`golden_answers` 生成 `train.parquet` 和 `test.parquet`。一条记录的逻辑结构如下（答案仅为示意）：

```python
{
    "data_source": "nq",
    "prompt": [{
        "role": "user",
        "content": "Answer the given question. ... Question: Who wrote Pride and Prejudice?"
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["Jane Austen"]}
    },
    "extra_info": {"split": "train", "index": 0}
}
```

`prompt` 包含交互协议：模型可以输出 `<think>...</think>`、`<search>查询词</search>`，读取检索器注入的 `<information>...</information>`，最后用 `<answer>答案</answer>` 作答。`ground_truth.target` 用于训练和验证时的规则评分，不拼进模型的正常提问。数据集没有预先标注的搜索轨迹；轨迹由当前策略在线生成。

### 1.2 进入训练器后的批次和轨迹

`main_ppo.py` 创建 `RLHFDataset` 和 dataloader，`RayPPOTrainer.fit()` 取一个 batch 后包装为 `DataProto`。每条问题得到一个 `uid`；按 `actor_rollout_ref.rollout.n` 扩展时，同一问题的多条采样轨迹保留同一个 `uid`，供 GRPO 分组。

Agent rollout 可能形成如下完整响应：

```text
<think>需要查证作者</think><search>Pride and Prejudice author</search>
<information>Doc 1(Title: ...) ...</information>
<think>资料给出了作者</think><answer>Jane Austen</answer>
```

veRL 训练批次中的主要字段如下。`[B, R]` 表示 batch 大小为 `B`、填充后的 response 长度为 `R`。

| 字段 | 内容及来源 |
| --- | --- |
| `prompts`, `input_ids`, `attention_mask`, `position_ids` | token 化的原始提示，以及“提示 + 完整轨迹”的模型输入、有效位和位置。检索文本在 `input_ids` 中，供后续生成读取。 |
| `responses` | 拼接后的完整轨迹，既含策略生成 token，也含检索观察 token；形状 `[B, R]`。 |
| `info_mask` | 与完整输入对齐：策略生成内容为有效位，检索观察和 padding 为 0。生成管理器通过把观察位置替换为 pad id 后构造此掩码。 |
| `response_mask` | 从 `attention_mask` 截取响应部分；包括有效的检索观察，排除 padding。 |
| `loss_mask` | 从 `info_mask` 截取响应部分；只保留策略生成 token，排除检索观察与 padding。 |
| `old_log_probs`, `ref_log_prob` | 当前采样策略的旧 log 概率，以及启用参考策略时的参考 log 概率；均按响应 token 对齐。 |
| `token_level_scores`, `token_level_rewards` | `[B, R]` 的 token 奖励。规则评分通常只写在最后一个有效响应 token；若启用 reward KL 惩罚，后者还会包含该惩罚。 |
| `values`, `advantages`, `returns` | PPO/GAE 使用 Critic 的逐 token value；GRPO 不需要 `values`。两种算法均产生优势和 return。 |
| `uid`, `reward_model`, `data_source`, `extra_info` | 非张量元数据；分别用于分组、取标准答案、选择评分函数和保留样本信息。 |

这里有三个不同的“有效位”：`attention_mask` 表示模型可以看见的 token，`response_mask` 表示完整响应中的非 padding token，`loss_mask` 表示 Actor 应承担梯度的动作 token。`<information>` 对模型可见，但不是模型生成的动作，因此其 Actor loss 掩码为 0。

### 1.3 这些字段在哪里产生、在哪里使用

这些字段不是一次性从 Parquet 读出的，而是在“预处理 → 读取问题 → 在线生成轨迹 → 计算训练信号 → 更新模型”过程中逐步加入 `DataProto`：

| 阶段 | 代码位置 | 产生或使用的字段 |
| --- | --- | --- |
| 写入数据集 | `scripts/data_process/nq_search.py` 的 `process_fn()` | 把 `prompt`、`reward_model.ground_truth`、`data_source`、`extra_info` 写入 Parquet。 |
| 读取并编码问题 | `verl/utils/dataset/rl_dataset.py` 的 `RLHFDataset.__getitem__()` | 将 `prompt` 转为初始 `input_ids`、`attention_mask`、`position_ids`；标准答案仍保存在非张量字段中。 |
| 拼接多轮轨迹 | `verl/llm_agent/generation.py` 的 `run_llm_loop()`、`_update_right_side()` 和 `_compose_final_output()` | 每轮拼接模型输出与检索观察，形成 `responses`；最终组装 `prompts`、完整 `input_ids`、`attention_mask`、`position_ids` 和 `info_mask`。 |
| 整理训练批次 | `verl/trainer/ppo/ray_trainer.py` 的 `fit()`、`compute_response_mask()` | 给每道题分配 `uid`，按采样数扩展并合并轨迹，从 `attention_mask` 取出 `response_mask`。 |
| 计算概率、奖励和优势 | `ray_trainer.py` 的 `fit()`，以及 `verl/workers/reward_manager/naive.py` | Actor worker 生成 `old_log_probs`；按配置计算 `ref_log_prob`、`values`；reward manager 把规则分数写到最后一个有效响应 token；训练器写入 `token_level_scores`、`token_level_rewards`、`advantages`、`returns`。 |
| 计算 Actor loss | `ray_trainer.py` 的 `_create_loss_mask()`，`verl/workers/actor/dp_actor.py` 的 `update_policy()` | 从 `info_mask` 截出 `loss_mask`，用于 Actor 的 policy loss，以及启用时的熵项和 KL loss。 |

`info_mask` 的形成需要特别区分“模型实际读到的内容”和“用于标记来源的副本”。`generation.py` 的 `_info_masked_concatenate_with_padding()` 将检索返回的 `<information>...</information>` 保留在真实 `responses` 中；在另一个副本 `responses_with_info_mask` 中，把观察位置改成 pad id。`_compose_final_output()` 分别据此生成 `attention_mask` 和 `info_mask`。因此模型下一轮可以读到检索文本，而 Actor 更新时这些观察 token 对应的 `loss_mask` 为 0。

上表列的是可能出现的字段，不表示每种配置都有全部字段：当前 PPO 默认不启用参考策略，因此不产生 `ref_log_prob`；GRPO 不使用 Critic，因此不产生 `values`。

## 2. 奖励如何计算

当前训练使用**规则奖励**，没有训练一个神经网络 Reward Model。可以把奖励流程分为两层：`NaiveRewardManager` 负责从一条轨迹中取出模型回答和参考答案，并把分数放入奖励张量；具体评分函数负责判断答案是否正确。

调用顺序如下：

1. `verl/trainer/main_ppo.py` 的 `TaskRunner.run()` 调用 `load_reward_manager()`，分别创建训练用的 `reward_fn` 和验证用的 `val_reward_fn`。`verl/trainer/ppo/reward.py` 的 `load_reward_manager()` 按 `config.reward_model.reward_manager` 选择管理器；当前默认是 `NaiveRewardManager`。由于 `custom_reward_function.path` 为空，传入管理器的评分函数是 `default_compute_score()`。
2. `verl/trainer/ppo/ray_trainer.py` 的 `fit()` 在生成并合并一批轨迹后，调用 `compute_reward(batch, self.reward_fn)`，从而执行 `NaiveRewardManager.__call__()`。后者逐条解码 `responses`，从该条样本的非张量字段取 `reward_model.ground_truth` 和 `data_source="nq"`，再调用 `self.compute_score(data_source=..., solution_str=..., ground_truth=...)`。这里的 `self.compute_score` 就是上一步传入的 `default_compute_score()`。
3. `verl/utils/reward_score/__init__.py` 的 `default_compute_score()` 按 `data_source` 分发。`"nq"` 命中 Search-R1 问答分支，于是调用 `verl/utils/reward_score/search_r1_like_qa_em.py` 的 `compute_score(solution_str, ground_truth)`。`data_source` 用来选择评分器，本身不参与答案匹配。
4. NQ 评分器从完整响应中提取最后一个 `<answer>...</answer>`，与 `ground_truth["target"]` 中的可接受答案比较，返回标量分数。
5. `NaiveRewardManager` 创建与 `responses` 同形状的零张量，把标量分数写在该轨迹最后一个有效响应 token 的位置；其余位置保持 0。训练器将其存为 `token_level_scores`，供后续优势计算使用。

例如标准答案为 `{"target": ["Jane Austen"]}`，模型最后输出 `<answer>Jane Austen</answer>`，这条轨迹的规则分数通常是 1。也可以通过 `custom_reward_function` 配置替换评分函数，但当前脚本没有这样做。

评分器从完整响应中提取**最后一个** `<answer>...</answer>`。它将预测和 `ground_truth.target` 中任一可接受答案归一化后做 exact match：转小写、去标点、去冠词、规范空白。无答案标签或不匹配得 0；匹配通常得 1。若输出了超过 10 个开或闭 `answer` 标签，匹配分降为 0.25。默认 `format_score=0`，搜索次数和检索分数不直接进入该评分。

在当前 `algorithm.use_kl_in_reward=false` 配置下，`token_level_rewards` 与 `token_level_scores` 相同。这里的 `reward_model.style="rule"` 是数据中的评分方式标记，不意味着启动了 Reward Model worker。

## 3. 当前采用的 RL 算法

两个根目录脚本共用 `python -m verl.trainer.main_ppo` 和 `RayPPOTrainer`；入口名称不表示 GRPO 也会训练 Critic。

| 配置 | PPO：`scripts/ppo/train.sh` | GRPO：`scripts/grpo/train.sh` |
| --- | --- | --- |
| `algorithm.adv_estimator` | `gae` | `grpo` |
| 每题采样数 `rollout.n` | 默认 1 | 4 |
| Critic / `values` | 启用；预测 value、计算 GAE 并更新 Critic | 默认关闭；不计算 value，也不更新 Critic |
| 优势计算 | `compute_gae_advantage_return()`：用末端奖励、value、`gamma`、`lam` 计算逐 token advantage/return | `compute_grpo_outcome_advantage()`：先求每条轨迹的总奖励，再按相同 `uid` 分组，计算 `(score - 组均值) / (组标准差 + ε)` 并广播到响应有效位 |
| 参考策略 | 当前默认不创建；`use_kl_in_reward=false` 且未启用 Actor KL loss | 启用 Actor KL loss，因此创建参考策略并计算 `ref_log_prob` |
| Actor 目标 | 复用 veRL 的 PPO clipped policy loss | 同样复用 veRL 的 policy loss，使用 GRPO 优势，并加配置的参考策略 KL loss |

GRPO 中某一组奖励完全相同时，组内中心化优势为 0。当前实现的优势计算本身复用 veRL `core_algos.py`，**Search-R1 没有另写 GAE 或 GRPO 公式**；关键是让它们接收多轮搜索轨迹和终局 EM 奖励。当前 `compute_advantage()` 使用 `response_mask`，并不是用 `loss_mask` 重新定义优势；只对策略动作求梯度发生在后续 Actor loss 阶段。

## 4. veRL 中的完整训练过程

### 4.1 模型与工作进程初始化

`scripts/ppo/train.sh` / `scripts/grpo/train.sh` 通过 Hydra 传入模型、数据、算法与检索配置；`main_ppo.py` 初始化 Ray、tokenizer、数据集、资源池和 `RayPPOTrainer`。`init_workers()` 使用 veRL 的 FSDP/Ray worker 初始化 Actor + vLLM Rollout；PPO 配置创建 Critic；只有配置需要 KL 时才创建 Reference Policy；仅配置了学习式奖励模型时才创建 Reward Model worker。当前规则奖励在 trainer 进程的 reward manager 中执行。

这部分主要沿用 veRL。Search-R1 接入本地 Qwen、NQ 和检索配置，并在训练器中创建 `LLMGenerationManager` 供搜索模式使用。

### 4.2 取 batch 与 Agent rollout

`RayPPOTrainer.fit()` 从 dataloader 取问题，给样本分配 `uid`，再按 `rollout.n` 展开。普通路径调用 `actor_rollout_wg.generate_sequences()` 得到一次生成；`do_search=true` 时，Search-R1 改为调用 `verl/llm_agent/generation.py` 的 `run_llm_loop()`。该管理器仍调用 veRL/vLLM 的 `generate_sequences()` 生成**每一轮**模型输出，但负责跨轮循环：

1. 截断一轮输出到 `</search>` 或 `</answer>`，解析动作。
2. 遇到 `<search>` 时向检索服务 `/retrieve` 发送查询，将 top-k 文档包装成 `<information>...</information>`；遇到 `<answer>` 时结束该轨迹。
3. 将生成文本和观察拼接回上下文，让 Actor 继续生成；无效动作会得到纠错提示。受 `max_turns`、响应长度和上下文长度限制；最后一次生成不再执行真实检索。
4. 输出完整 `DataProto`，并构造 `info_mask`，记录哪些响应 token 来自环境观察。

当前脚本设置 `max_turns=2`、`retriever.topk=3`。这一步是普通单轮 RL 与 Agentic RL 最主要的接口差异：训练样本仍是一道问题，但一次 rollout 可以包含多次“动作 → 外部观察 → 后续动作”。

### 4.3 概率、奖励、价值和优势

搜索路径在多轮轨迹拼好后调用 `actor_rollout_wg.compute_log_prob()` 得到 `old_log_probs`。之后 trainer 将原始问题字段与 rollout 合并，构造 `response_mask`，由规则 reward manager 算奖励。若启用参考策略，再计算 `ref_log_prob`；若启用 Critic，再计算 `values`。这些概率与 value 的 worker 接口主要复用 veRL，无需为搜索重新实现模型加载或前向计算。

训练器依据配置选择 GAE 或 GRPO 算优势：PPO 使用稀疏终局奖励、Critic value 和 GAE；GRPO 根据同一 `uid` 的四条轨迹做组内比较。当前两条路径都没有启用 reward 内 KL 惩罚；GRPO 的 KL 作用于 Actor loss。

### 4.4 Actor loss 与参数更新

更新 Actor 前，Search-R1 的 `_create_loss_mask()` 从 `info_mask` 截取响应部分并生成 `loss_mask`。`verl/workers/actor/dp_actor.py` 在 `state_masking=true` 时把它传给 veRL 的 policy loss 聚合函数；若启用熵或 Actor KL loss，这些项也使用同一掩码。因此 `<think>`、`<search>`、`<answer>` 等由策略生成的有效 token 可以参与 Actor 优化，而 `<information>` 观察和 padding 不参与 Actor 梯度。

随后复用 veRL 的 `update_actor()` 完成 mini-batch/micro-batch、前向、clipped policy loss、反向传播和优化器更新。PPO 还会调用 `update_critic()`；当前 Critic 路径仍使用 `response_mask`，不要把 Actor 的 `loss_mask` 误写成所有训练计算的统一掩码。下一轮 rollout 由 veRL 的 Actor/Rollout 协调机制使用更新后的策略权重。训练器还负责验证、日志及检查点保存。

## 5. 模块与职责对照

| 模块 | 在当前项目中的职责 | Search-R1 接入点 |
| --- | --- | --- |
| `scripts/data_process/nq_search.py` | 将 NQ 转成含 prompt、标准答案和元数据的 Parquet | 定义搜索提示词及数据格式 |
| `verl/trainer/main_ppo.py` | Hydra/Ray 入口、worker、数据集及 reward manager 初始化 | 基本复用 |
| `verl/trainer/ppo/ray_trainer.py` | 取 batch、调度 rollout、概率、奖励、价值、优势、更新与保存 | `do_search` 分支、多轮生成管理器、`loss_mask` 构造 |
| `verl/llm_agent/generation.py` | 多轮动作解析、检索调用、观察拼接和 `info_mask` | Agentic rollout 的核心实现；实际入口是此目录，不是根目录的同名文件 |
| `search/retrieval_server.py` | E5 + FAISS 检索 Wikipedia，提供 `/retrieve` | 外部环境，不参与梯度 |
| `verl/workers/reward_manager/naive.py`、`verl/utils/reward_score/search_r1_like_qa_em.py` | 将完整响应转换为末端 EM 奖励 | 复用 reward manager 接口，使用 NQ 规则评分 |
| `verl/trainer/ppo/core_algos.py` | GAE、GRPO 优势及 policy loss 算法 | 现有算法实现，无需另写优势公式 |
| `verl/workers/actor/dp_actor.py` | 按优势计算 Actor loss 并更新策略 | `state_masking` 时使用 `loss_mask` 排除环境观察 |

**当前复现状态：**本地 PPO 已完成 100 步训练并保存检查点；GRPO 仅做过单步 smoke test。本文描述 GRPO 的代码路径和脚本配置，不将其表述为已完成的正式训练结果。
