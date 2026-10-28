# Search-R1 论文主实验配置（Qwen2.5-3B Base）

本实验使用原作者发布的 `PeterJinGo/nq_hotpotqa_train`，固定版本
`b7d80abfee334a7a91cb377544f09180d58b34f6`。训练集为 NQ 79,168 条与
HotpotQA 90,447 条；`test.parquet` 含七个 QA 数据集共 51,713 条。下载文件放在
`/home/jk/work/datasets/nq_hotpotqa_searchr1_paper`。

训练中的周期验证使用每数据集随机种子 42 抽取至多 500 条的固定子集
（Bamboogle 全部 125 条，共 3,125 条）；论文表格需在训练完成后对完整
`test.parquet` 单独评测，不能把周期验证子集的分数当作论文结果。

## 配置与资源

运行 `bash scripts/experiments/run_paper_reproduction.sh ppo` 或 `grpo`。脚本默认让
8 张 A100 参与训练；可以用 `CUDA_VISIBLE_DEVICES` 指定最多 8 张。
本地 E5 检索器使用 GPU 7，与第八个训练 worker 共享该卡。若 GPU 7
显存不足，可令训练只使用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`，
检索器仍使用 GPU 7。训练前启动 `bash scripts/retrieval/start_local.sh`。
在这台双 NUMA 机器上，8 卡时脚本默认设置 `NCCL_P2P_DISABLE=1`
和 `NCCL_IB_DISABLE=1`；初次短跑的跨组 FSDP 初始化曾停滞，设置后
可以进入训练循环。

论文主实验参数：500 步、动作预算 4、GRPO 每题 5 条轨迹、batch 512、
mini batch 256、top-3 检索、每 100 步保存检查点。
当前优化后的 GRPO 使用每 GPU Actor 与参考 log 概率 micro batch 4，
rollout `max_num_seqs=32`、`max_num_batched_tokens=8192`。它与论文的
总 micro batch 64 不同，需在结果中注明。PPO 与 GRPO 分别写入新的检查点目录，
避免自动续跑旧的 NQ-only 100 步实验。

`max_turns=4` 在本项目生成器中表示最多四次带检索执行的生成循环；
若仍未结束，还会做一次不执行检索的最终生成。滚动上下文及完整响应
分别受 `data.max_prompt_length=4096` 限制。短跑时要检查截断、
`<answer>` 完整性、有效检索次数、显存和奖励。
8 卡短跑已经记录到若干 511–550 token 的检索观察，被
`data.max_obs_length=500` 截断。该设置与原训练脚本一致，
但截断比例和对 EM 的影响仍需从更多轨迹检查；本地检索语料与索引也会
影响论文结果的可比性。

2026-10-24 的 8 卡 GRPO 单步短跑使用 batch 8、每题 5 条轨迹：
`training/global_step=1`，40 条轨迹中训练奖励均值 0.05，
`response_length/clip_ratio=0.025`，峰值保留显存约 59.1 GiB/卡。
训练批次耗时 283.5 秒，小样本七项验证另耗时 148.5 秒。
曾出现 1420 token 的检索观察被裁到 500 token。
这些数字不能预测 512 batch 的峰值和吞吐；正式训练前还需在目标
检索配置下用更大的 batch 再测。

CPU 平坦 FAISS 索引约 64.6 GB，短跑中的单次检索耗时约 9–64 秒。
独立 GPU 索引已通过测试，可用
`RETRIEVER_FAISS_GPU=1 RETRIEVER_GPU=7 bash scripts/retrieval/start_local.sh`
将索引放在 GPU 7，并用 `CUDA_VISIBLE_DEVICES=0,1,2,3`
训练。分块上传后的索引占用约 33 GiB；32 条查询的接口请求约 0.2 秒。
GPU 索引和 Actor 在同一张 80 GB A100 上会超过短跑所见的显存峰值，
因此 GPU 7 必须独立于训练卡。4 卡仍符合最多 8 卡约束，但训练并行度
与论文的 8 张 H100 不同。

4 卡、GPU 检索的第二轮单步短跑使用 batch 32 和组大小 5，
`training/global_step=1`，峰值保留显存约 66.9 GiB/卡，
`response_length/clip_ratio=0.00625`，训练步骤约 441.2 秒，
其中生成 184.4 秒、Actor 更新 186.8 秒。正式 batch 512 的峰值
尚未测得，仍存在长轨迹触发 OOM 的风险。

原 4 卡正式 GRPO 在第 2 步停止，按用户要求改做独立短跑并从头重启。
短跑均使用相同的 32 条训练样本，组大小 5、4 轮；以下时间仅含训练步骤，
不含模型加载。各配置的随机生成轨迹不完全相同，因此比较结果为近似值。

| 训练配置 | 单步时间 | 峰值预留显存/卡 | 生成 token 总量 |
| --- | ---: | ---: | ---: |
| 4 卡，Actor/ref micro 1，并发 16 | 436 秒 | 66.9 GiB | 231,545 |
| 4 卡，Actor micro 4、ref micro 1，并发 16 | 303 秒 | 70.6 GiB | 231,545 |
| 4 卡，Actor/ref micro 4，并发 16 | 259 秒 | 70.6 GiB | 231,545 |
| 4 卡，Actor/ref micro 4，并发 32 | 220 秒 | 67.8 GiB | 218,212 |
| 5 卡，Actor/ref micro 4，并发 16 | 287 秒 | 69.0 GiB | 226,229 |

选中配置另外使用固定 128 条样本压测，640 条候选轨迹的一整步耗时
676 秒，峰值预留显存 71.0 GiB/卡，未发生 OOM。batch 512 的峰值
尚未直接测得，后续仍须监控。短跑结束时 Ray worker 偶发段错误，
但步骤已完成并输出指标；正式进程的稳定性仍须观察。

2026-10-24 已从头启动优化后的正式 GRPO：GPU 0–3 训练、GPU 7 检索，
500 步，初始验证关闭，周期验证使用 3,125 条固定子集。以
`logs/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized.log` 的步数和
`verl_checkpoints/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized` 的检查点
确认进度。完整七项评测、PPO 正式训练尚未完成。

2026-10-26 按用户要求准备在第 100 步保存后停止，以便关机后续训。
第 52 步时没有磁盘检查点；短暂挂起后已恢复原进程。独立守护脚本
`scripts/stop_after_paper_checkpoint.sh` 检查第 100 步的 dataloader、
4 个 rank 的模型与优化器文件，以及 `latest_checkpointed_iteration.txt`，
确认完整后停止训练进程组。其结果写入
`logs/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized-stop-at-100.log`。
下次使用同一实验目录和 `trainer.resume_mode=auto` 可从该检查点继续。
若关机后从第 100 步或后续检查点恢复，使用
`bash scripts/resume_paper_grpo_after_100.sh`。脚本先检查最新的完整
检查点（至少第 100 步），再用同一批训练参数继续到 500 步。续训设置
`trainer.save_freq=15` 和 `trainer.save_freq_offset=100`，因此在
第 115、130、145、……步保存；从第 100 步续训后，固定验证子集
每 10 步评测一次（110、120、130、……）。
续训还设置 `trainer.max_complete_ckpt_to_keep=1`：新 checkpoint
完整写入并更新最新步数标记后，删除旧检查点目录。保存过程中临时需要
约两份检查点的磁盘空间。

## SwanLab 记录

正式 GRPO 使用 W&B 离线日志。2026-10-26 已将当时可读取的第 1–52 步
导入 `swanlog/paper-grpo-optimized-offline-snapshot`；它只是一个指标快照，
不会自动跟随训练更新。该快照已上传到 SwanLab：
`https://swanlab.cn/@xujintao/Search-R1/runs/b3vr57ri`。
需要重新同步同一离线快照时可运行：

```bash
/home/jk/work/envs/searchr1-verl/bin/swanlab sync \
  swanlog/paper-grpo-optimized-offline-snapshot/run-20261026_082725-93geox9j \
  --id b3vr57ri
```

训练到第 100 步并停止后，应从最新 W&B 离线文件重新导入完整记录，
再用 `swanlab sync --id b3vr57ri` 补充到同一云端实验。导入命令需设置 `--mode offline`，且不应补入
其他实验的最终 EM。

## 数据与运行检查

```bash
/home/jk/work/envs/searchr1-verl/bin/python scripts/prepare_paper_validation.py \
  --data-dir /home/jk/work/datasets/nq_hotpotqa_searchr1_paper
bash scripts/experiments/run_paper_reproduction.sh grpo --cfg job
```

短跑可在命令末尾添加 Hydra 覆盖项，使用单独的实验目录，例如
`trainer.total_training_steps=1`、`data.train_batch_size=8`、
`actor_rollout_ref.actor.ppo_mini_batch_size=8`、
`data.val_files=/home/jk/work/datasets/nq_hotpotqa_searchr1_paper/validation_smoke.parquet`、
`trainer.experiment_name=paper-grpo-smoke`、
`trainer.default_local_dir=verl_checkpoints/paper-grpo-smoke`。
短跑结果只用于检查训练链路。

当前 `scripts/ppo/eval.sh` 和 `scripts/grpo/eval.sh` 固定指向 NQ 旧模型、旧目录和
`max_turns=2`，不能直接用于论文七项评测。完整评测需先将本次训练的
Actor 检查点合并为 Hugging Face 格式，再运行：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir verl_checkpoints/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized/global_step_500/actor \
  --target_dir models/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized
bash scripts/experiments/eval_paper_reproduction.sh \
  "$PWD/models/paper-nq-hotpotqa-qwen2.5-3b-grpo-optimized"
```

评测脚本使用完整 `test.parquet`、`max_turns=4` 和同一检索器，
终端输出七个数据源的 EM。PPO 模型将上述路径中的 `grpo` 换成 `ppo`。
