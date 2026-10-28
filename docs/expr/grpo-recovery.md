# GRPO 持久运行与恢复

启动：`bash scripts/grpo/start_paper_durable.sh`。

- 配置入口：`scripts/grpo/run_paper_durable.sh`，7 卡，train/mini batch 336，Ref/old-log-prob/Actor micro batch 为 80/80/16，vLLM 显存比例 0.75、并发序列 96、批处理 token 上限 16384。Actor 每卡 mini batch 为 48，须能被 micro batch 整除，因此不能设为 18。
- 训练到 500 步；100 步不停止，每 100 步验证。
- 每 5 步保存，保留最近 2 份检查点；`resume_mode=auto` 从最近保存步恢复。
- 持久 user systemd 服务：`search-r1-grpo-7gpu-b336`、`search-r1-retriever`，均启用开机启动，用户 Linger 已启用。
- GPU 分配在 `scripts/systemd/` 中按 UUID 固定，检索使用独立 GPU。训练启动时检查 8 张卡均存在，再按 PCI 顺序将 UUID 转为当前 vLLM 支持的整数编号。
- 普通失败 60 秒后重启服务，2 小时内最多启动 3 次。OOM 不触发整机重启，因为重启不能解决配置过大的问题。
- `scripts/grpo_cuda_recovery.py` 只检查本次启动日志。遇到 unknown CUDA error、device lost、unspecified launch failure、GPU fallen off bus、uncorrectable ECC 等致命 GPU 错误时请求无交互重启。
- 内核报致命 CUDA 错误，或训练失败且 GPU 数量少于 8 时，自动请求整机重启；6 小时内最多自动重启 2 次，避免无限重启。权限拒绝会写入 journal。已实际执行过一次手动重启并确认服务开机自启。
- 手工正常停止不会触发重启。停止训练：`systemctl --user stop search-r1-grpo-7gpu-b336`。跨重启也停用：`systemctl --user disable --now search-r1-grpo-7gpu-b336`。
- 查看状态：`systemctl --user status search-r1-grpo-7gpu-b336`；恢复日志：`journalctl --user -u search-r1-grpo-7gpu-b336`。
- GPU 温度、功耗和显存每 15 秒记录到 `logs/grpo-gpu-thermal.csv`，用于核对 GPU4 故障前的峰值。

2026-10-27 恢复起点为第 10 步；旧运行完成过第 13 步，第 14 步报 CUDA unknown error，未保存的步骤需要重算。当前机制不保存故障瞬间的内存状态。
