"""Print comparable timing and memory metrics from one-step benchmark logs."""

import re
from pathlib import Path


KEYS = (
    "timing_s/gen",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/step",
    "perf/total_num_tokens",
    "perf/max_memory_reserved_gb",
    "response_length/mean",
    "response_length/clip_ratio",
)


def main() -> None:
    print("variant\t" + "\t".join(KEYS))
    for path in sorted(Path("logs").glob("paper-benchmark-*.log")):
        if path.stem.endswith("-launch"):
            continue
        text = path.read_text(errors="replace")
        metric_lines = [line for line in text.splitlines() if "step:1 - global_seqlen/min" in line]
        if not metric_lines:
            continue
        line = metric_lines[-1]
        values = []
        for key in KEYS:
            match = re.search(r"(?:^| - )" + re.escape(key) + r":([^ ]+)", line)
            values.append(match.group(1) if match else "NA")
        print(path.stem.removeprefix("paper-benchmark-") + "\t" + "\t".join(values))


if __name__ == "__main__":
    main()
