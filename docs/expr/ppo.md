# NQ PPO 复现实验

本地 `global_step_100` checkpoint 对应已完成的 **NQ 单数据集 PPO** 实验。以下配置以[实际训练日志](../../logs/train_ppo.log)和[启动脚本](../../train_ppo.sh)为准。

| 项目 | 实际配置 |
| --- | --- |
| 基座模型 | Qwen2.5-3B Base；Actor、Critic 均从该模型初始化 |
| 数据 | NQ 训练集 **79,168** 条，测试集 **3,610** 条 |
| 算法 | PPO＋GAE，使用 Critic；`gamma=1`、`lam=1` |
| 训练规模 | **4 张 GPU**；batch **512**、mini batch **256**、Actor/Critic 每卡 micro batch **1**；总计 **100 步** |
| Rollout | 每题采样 **1** 条轨迹；`max_turns=2`；本地 E5＋FAISS 检索，`topk=3` |
| 长度限制 | prompt **4096**、单次生成 **500**、检索观察 **500** token |
| 学习率 | Actor `1e-6`；Critic `1e-5` |
| 奖励与掩码 | NQ 答案的规则 EM 评分；Actor 只对模型生成的 token 计算损失；固定 reward KL 系数 `0.001` |
| 验证与保存 | 训练前验证，此后每 **50** 步验证；第 **100** 步保存 checkpoint |

## 复现效果

指标为 NQ 测试集上的 `val-core/nq/reward/mean@1`（规则 EM 奖励均值）。

| 阶段 | NQ 验证 EM |
| --- | ---: |
| 训练前 | **3.66%** |
| 第 50 步 | **35.01%** |
| 第 100 步 | **34.60%** |

第 100 步比训练前提高约 **30.94 个百分点**。训练正常完成 `100/100` 步，退出码为 `0`；checkpoint 位于 [`global_step_100`](../../verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-em/global_step_100)。第 0、50 步指标见[离线 W&B 记录](../../wandb/offline-run-20261002_233647-6qpkbt86/run-6qpkbt86.wandb)，第 100 步指标见[训练日志](../../logs/train_ppo.log)中的 `Final validation metrics`。

这是 **NQ 内置验证**的结果；尚无该 checkpoint 的七项 benchmark 评测结果。

[SwanLab 实验记录](https://swanlab.cn/@xujintao/Search-R1/runs/snbydrrf)已导入离线 W&B 的逐步指标，并补入训练日志中的第 100 步验证值。
