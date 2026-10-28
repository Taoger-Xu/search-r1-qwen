#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "$HOME/.config/systemd/user"
cp scripts/systemd/search-r1-retriever.service scripts/systemd/search-r1-grpo-7gpu-b336.service "$HOME/.config/systemd/user/"
systemctl --user stop search-r1-grpo-7gpu-stop100.service 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user reset-failed search-r1-grpo-7gpu-b336.service 2>/dev/null || true
systemctl --user enable --now search-r1-retriever.service search-r1-grpo-7gpu-b336.service
