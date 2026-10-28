"""Reboot after a fatal CUDA failure; ordinary failures use systemd restarts."""
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'logs/paper-nq-hotpotqa-qwen2.5-3b-grpo-7gpu-b336.log'
STAMP = ROOT / 'logs/grpo-last-auto-reboot'


def fatal_cuda(text):
    return any(message in text.lower() for message in (
        'cuda error: unknown error', 'cuda error: device is lost',
        'cuda error: unspecified launch failure', 'gpu has fallen off the bus',
        'cuda error: uncorrectable ecc error',
    ))


def gpu_missing():
    try:
        output = subprocess.check_output(['nvidia-smi', '-L'], text=True, timeout=15)
        return sum(line.startswith('GPU ') for line in output.splitlines()) < 8
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return True


def main():
    if os.environ.get('SERVICE_RESULT', 'success') == 'success':
        return
    offset_file = ROOT / 'logs/grpo-attempt-offset'
    if not LOG.exists() or not offset_file.exists():
        return
    with LOG.open('rb') as stream:
        stream.seek(int(offset_file.read_text().strip()))
        failure = stream.read().decode(errors='replace')
    if not fatal_cuda(failure) and not gpu_missing():
        print('No fatal CUDA device error: using bounded systemd service retries.', flush=True)
        return
    now = time.time()
    history = STAMP.read_text().splitlines() if STAMP.exists() else []
    recent = [float(value) for value in history if now - float(value) < 6 * 3600]
    if len(recent) >= 2:
        print('Automatic reboot suppressed: two reboots within six hours.', flush=True)
        return
    STAMP.write_text('\n'.join(map(str, recent + [now])) + '\n')
    print('Fatal CUDA error: requesting machine reboot; enabled services will resume from checkpoint.', flush=True)
    result = subprocess.run(['systemctl', '--no-ask-password', 'reboot'], timeout=30)
    if result.returncode:
        STAMP.write_text('\n'.join(map(str, recent)) + '\n')
        raise RuntimeError('Automatic reboot denied or failed; inspect systemd journal.')


if __name__ == '__main__':
    main()
