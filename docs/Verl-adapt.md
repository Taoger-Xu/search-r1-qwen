# veRL 如何适配 Search-R1

## 1. 文档范围

Search-R1 是在 veRL 分布式 PPO/GRPO 训练框架上的二次开发。veRL 负责 Ray 调度、模型生命周期、批处理、优势计算、PPO 优化、FSDP/vLLM 参数同步和 checkpoint；Search-R1 主要改造 Agent rollout、检索环境交互、答案奖励以及 token-level loss mask。

本文只关注强化学习主流程，不展开 Ray、FSDP 和检索服务的工程细节。

## 2. 完整训练闭环

```text
加载 tokenizer 和模型
        ↓
Ray 初始化 Actor / Rollout / Reference / Critic Worker
        ↓
采样问题 prompt
        ↓
generate_sequences：模型生成 + 检索器多轮交互
        ↓
RewardManager：解析最终 answer，计算 EM reward
        ↓
Critic 计算 value
        ↓
PPO GAE（或 GRPO group-relative）计算 advantage / return
        ↓
Actor 根据 response_mask 和 loss_mask 计算 PPO loss
        ↓
Critic 拟合 return
        ↓
反向传播和参数更新
        ↓
Actor 权重同步到 vLLM rollout
        ↓
进入下一个 training step
```

veRL 的训练骨架位于 `verl/trainer/ppo/ray_trainer.py`。Search-R1 的实际 Agent 生成逻辑位于 `verl/llm_agent/generation.py`，奖励逻辑位于 `verl/workers/reward_manager/` 和 `verl/utils/reward_score/`。

## 3. 模型加载和 Worker 初始化

### 3.1 Ray 中的角色

训练启动后，veRL 通过 Ray 创建并调度不同角色：

```text
Ray
 ├── ActorRolloutRefWorker
 │    ├── Actor policy model
 │    ├── Rollout engine（vLLM）
 │    └── Reference model
 ├── CriticWorker
 │    └── Value model
 └── RewardManager
      └── Search-R1 reward calculation
```

这些角色可以映射到不同 GPU 或 GPU 资源池。Search-R1 使用 veRL 的 WorkerGroup 和资源调度机制，不重新实现 PPO 的分布式执行框架。

### 3.2 `init_workers` 的职责

训练器初始化 Worker 后，主要完成以下工作：

1. 创建 Actor、Rollout、Reference 和 Critic Worker；
2. 加载 tokenizer 和 Qwen2.5-3B；
3. 初始化 Actor policy；
4. 初始化 Reference policy，用于旧策略概率或 KL 约束；
5. 初始化 Critic/value model；
6. 初始化 vLLM rollout engine；
7. 初始化 FSDP、optimizer 和 scheduler；
8. 建立 Ray WorkerGroup；
9. 根据训练配置决定是否启用 reference KL、参数 offload 和 gradient checkpoint。

Search-R1 默认使用 Qwen2.5-3B 作为 Actor 和 Reference。Reward model 不是额外的大模型，而是一个规则型答案评分器：从 `<answer>` 中提取答案，与标准答案进行 EM 比较。

## 4. Search-R1 对 rollout 的改造

普通语言模型 PPO 的 rollout 是：

```text
prompt → model.generate() → response
```

Search-R1 将 response 扩展为带外部环境 observation 的 Agent trajectory：

```text
prompt
  ↓
模型生成 <think> 和 <search>
  ↓
解析 query，调用检索器
  ↓
返回 <information> 文档
  ↓
将 observation 拼接回上下文
  ↓
模型继续生成
  ↓
再次搜索或生成 <answer>
```

### 4.1 `generate_sequences`

`verl/llm_agent/generation.py` 中的生成管理器按 turn 循环调用 rollout 模型。抽象流程如下：

```python
def generate_sequences(prompts):
    sequences = initialize(prompts)

    for turn in range(max_turns):
        responses = rollout_model.generate(sequences)

        for sequence, response in zip(sequences, responses):
            if contains_search(response):
                query = parse_search_query(response)
                documents = retriever.search(query)
                sequence.append(response)
                sequence.append(format_documents(documents))
            elif contains_answer(response):
                sequence.finish(response)
            else:
                sequence.continue_generation()

    return sequences
```

例如：

```text
用户：Who wrote Pride and Prejudice?

模型：
<think>I need to identify the author.</think>
<search>Pride and Prejudice author</search>

环境：
<information>
Pride and Prejudice was written by Jane Austen.
</information>

模型：
<think>The retrieved passage identifies the author.</think>
<answer>Jane Austen</answer>
```

`<search>` 是模型产生的 action，`<information>` 是检索环境返回的 observation，`<answer>` 是最终 action/结果。达到 `<answer>`、`max_turns` 或最大 response 长度后，trajectory 结束。

### 4.2 检索器作为环境

Search-R1 的检索器可以抽象为 Agent 环境：

```text
Agent action：生成 search query
Environment：返回 top-k documents
Agent action：继续思考、再次搜索或回答
```

因此，检索结果不是模型参数产生的 token，而是 rollout 期间注入的外部状态。模型能够读取这些文档，但后面的策略 loss 不应该要求模型复现它们。

## 5. Trajectory 中保存的数据

一条样本不再只有 `prompt + response`，还会保存：

```text
prompt tokens
response tokens
attention_mask
response_mask
loss_mask
search observations
reward
values
advantages
returns
```

完整序列可能是：

```text
prompt
+ <think>...</think>
+ <search>...</search>
+ <information>...</information>
+ <think>...</think>
+ <answer>...</answer>
```

其中：

- `<think>`、`<search>`、`<answer>` 是策略生成内容；
- `<information>` 是环境 observation；
- `loss_mask` 记录哪些 token 可以参与 Actor 更新。

## 6. Reward：从最终回答得到 EM

默认奖励逻辑位于 `verl/utils/reward_score/search_r1_like_qa_em.py`，流程是：

```text
完整 trajectory
  ↓
解析最后一个 <answer>...</answer>
  ↓
答案归一化
  ↓
与 ground truth 比较
  ↓
得到 EM reward
```

归一化通常包括：

- 转小写；
- 删除标点；
- 删除冠词；
- 压缩空白；
- 比较预测答案和标准答案是否完全一致。

基本奖励为：

```python
reward = 1.0  # 正确
reward = 0.0  # 错误
```

在训练 batch 中，这个稀疏结果奖励通常放到 response 的最后一个有效 token 上：

```text
response tokens:  t1  t2  t3  ...  tn
reward:           0   0   0   ...  1
```

如果模型输出多个 answer，奖励管理器会按项目中的答案解析规则处理，训练仍然围绕最终答案是否正确展开。

## 7. Advantage 和 Return

Search-R1 没有改变 PPO 的核心优势估计，而是复用 veRL 的实现。

### 7.1 PPO：Critic + GAE

Critic 对每个 response token 预测状态价值：

```text
V(s1), V(s2), ..., V(st)
```

然后根据 reward 和 value 计算：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)
A_t = δ_t + γλ A_{t+1}
```

Search-R1 的 reward 通常是末端稀疏奖励，因此 GAE 会把最终结果向前传播到整条 trajectory：

```text
最终答案正确 → 相关动作的 advantage 倾向为正
最终答案错误 → 相关动作的 advantage 倾向为负
```

`return` 是 Critic 的训练目标，`advantage` 是 Actor PPO 更新的权重。

### 7.2 GRPO 分支

项目也保留 GRPO。GRPO 为同一个 prompt 采样多条 response，并在 group 内归一化 reward：

```text
prompt
 ├── response 1 → reward 0
 ├── response 2 → reward 1
 ├── response 3 → reward 1
 └── response 4 → reward 0
```

典型形式为：

```text
A_i = (r_i - mean(r_group)) / std(r_group)
```

GRPO 不依赖单独的 Critic。当前正式完成的训练使用 PPO，因此实际使用的是 Critic + GAE。

## 8. Search-R1 的关键改造：loss mask

### 8.1 为什么需要 mask

检索文档是环境返回的 observation，不是 Actor 生成的 action。如果直接对整段序列计算语言模型 loss，模型会被要求学习预测：

```text
<information>document text</information>
```

这会把环境输出错误地当成策略输出。

因此，Search-R1 使用 token-level `loss_mask`：

```text
模型生成 token       loss_mask = 1
检索 observation     loss_mask = 0
padding              loss_mask = 0
```

示意如下：

```text
tokens：
prompt | think | search | information | think | answer
mask：     0       1       1       0          1       1
```

### 8.2 `attention_mask` 和 `loss_mask` 的区别

这两个 mask 作用不同：

| Mask | 作用 | 检索文档的取值 |
|---|---|---:|
| `attention_mask` | 决定模型能否看到 token | 1 |
| `loss_mask` | 决定 token 是否参与梯度计算 | 0 |

因此 Search-R1 的设计是：

> 模型可以读取检索结果，但不需要学习复现检索结果。

### 8.3 Actor 的 PPO loss

veRL 的 PPO Actor loss 先计算新旧策略概率比：

```python
ratio = exp(new_log_prob - old_log_prob)

surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages

ppo_loss = -min(surr1, surr2)
```

Search-R1 在聚合 loss 时加入有效 token mask：

```python
policy_loss = masked_mean(
    ppo_loss,
    response_mask * loss_mask,
)
```

因此 Actor 只更新：

1. 策略实际生成的 response token；
2. 非 observation token；
3. 非 padding token。

prompt、检索文档和 padding 都不产生 Actor policy gradient。

Critic 则在有效 response 区域拟合 return：

```python
critic_loss = masked_mean(
    value_loss,
    response_mask,
)
```

## 9. 一次 PPO training step

### 第一步：采样 prompt

从 NQ search 数据集中取出一批问题。

### 第二步：多轮 rollout

Actor 生成思考和搜索动作，Search-R1 调用检索器，将文档作为 observation 注入上下文，直到生成 answer 或达到 turn/长度限制。

### 第三步：计算奖励

从 `<answer>` 提取最终答案，进行归一化并计算 EM。

### 第四步：得到 old log probability

保存 rollout 策略对其生成 action 的概率：

```text
log π_old(a_t | s_t)
```

### 第五步：Critic 计算 value

```text
V(s_t)
```

### 第六步：计算 advantage 和 return

PPO 使用 GAE 将末端 EM reward 传播到 trajectory。

### 第七步：更新 Actor

重新计算当前策略概率：

```text
log π_θ(a_t | s_t)
```

然后使用 PPO ratio、clipping、advantage 和 `response_mask * loss_mask` 更新策略。

### 第八步：更新 Critic

Critic 拟合 return，学习判断当前思考和搜索状态对最终答对的价值。

### 第九步：同步 rollout 权重

Actor 更新后的 FSDP 权重同步到 vLLM。下一轮 rollout 使用新的策略。

## 10. 代码改造边界

| 功能 | veRL 原有部分 | Search-R1 改造部分 |
|---|---|---|
| Ray 调度 | WorkerGroup、资源池、角色映射 | 基本复用 |
| 模型加载 | Actor、Reference、Critic、vLLM 初始化 | 使用本地 Qwen2.5-3B 配置 |
| PPO 更新 | ratio、clipping、optimizer、scheduler | 接入额外 `loss_mask` |
| Advantage | PPO GAE、return | 复用；GRPO 另有 group-relative 分支 |
| Rollout | 单轮 `generate` | `verl/llm_agent/generation.py` 多轮 Agent loop |
| 环境 | 无外部环境 | Search query → retriever → information |
| Reward | 通用 reward manager 接口 | NQ answer EM |
| 参数同步 | FSDP ↔ vLLM | 复用 |
| Checkpoint | veRL checkpoint 机制 | 复用 |

## 11. 与普通 veRL PPO 的本质区别

普通 veRL PPO：

```text
prompt → response → reward → PPO update
```

Search-R1：

```text
prompt
 → thought
 → search action
 → retrieval observation
 → thought
 → answer
 → EM reward
 → masked PPO update
```

Search-R1 没有重新定义 PPO 的核心数学形式，主要改变了三点：

1. rollout 从单轮文本生成变成 Agent 与检索环境的多轮交互；
2. reward 从普通结果评分变成最终答案 EM；
3. loss 从整段 response loss 变成只对策略生成 token 计算 loss。

最终可以概括为：

> veRL 提供分布式 PPO 的训练引擎，Search-R1 将语言模型 response 扩展为带外部 observation 的 Agent trajectory，并通过 observation-aware mask 将 PPO 更新限制在 Agent 自己产生的动作上。

## 附录：将已完成的 W&B 离线记录导入 SwanLab

本项目提供 `scripts/import_wandb_to_swanlab.py`，它只读取 W&B 的 `.wandb` 文件，不会重新启动训练。脚本会完成三件事：解析结构化 history、补入训练结束后由 veRL 打印的最终验证 EM、写出 CSV/JSON 并创建 SwanLab 本地实验。

```bash
cd /home/jk/work/Search-R1-Qwen3
/home/jk/work/envs/searchr1-verl/bin/python \
  scripts/import_wandb_to_swanlab.py \
  wandb/offline-run-20261002_233647-6qpkbt86/run-6qpkbt86.wandb \
  --out swanlog/searchr1-ppo-import \
  --project Search-R1 \
  --experiment ppo-qwen2.5-3b-imported \
  --final-step 100 --final-em 0.345983379501385
```

只有显式提供以下两个参数时，才会补入训练结束后由 veRL 打印的最终验证指标：

```bash
--final-step 100 --final-em 0.345983379501385
```

输出文件为：

```text
swanlog/searchr1-ppo-import/metrics.csv
swanlog/searchr1-ppo-import/metrics.json
swanlog/searchr1-ppo-import/run-*/
```

其中 CSV 适合用 pandas、Matplotlib 或 Excel 做二次分析。打开本地 SwanLab 看板：

```bash
/home/jk/work/envs/searchr1-verl/bin/swanlab watch \
  swanlog/searchr1-ppo-import
```

如果当前环境提示缺少 `swanboard`，安装看板扩展后再执行上面的命令：

```bash
uv pip install --python /home/jk/work/envs/searchr1-verl/bin/python \
  'swanlab[dashboard]'
```

脚本导入的是 W&B 已记录的逐 step 训练指标；最终验证 EM 是从 `logs/train_ppo.log` 的 `Final validation metrics` 补入的。由于这次训练只在 step 0、50、100 进行验证，其他 step 的验证 EM 不应插值后当作真实评估结果。
