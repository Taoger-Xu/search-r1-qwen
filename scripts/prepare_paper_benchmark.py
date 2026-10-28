"""Select a fixed, balanced training batch for throughput comparisons."""

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = Path("/home/jk/work/datasets/nq_hotpotqa_searchr1_paper")
SOURCE = DATA_DIR / "train.parquet"
DESTINATION = DATA_DIR / "benchmark_32.parquet"


def main() -> None:
    table = pq.read_table(SOURCE)
    sources = table.column("data_source").to_pylist()
    rng = random.Random(42)
    selected = []
    for source in ("nq", "hotpotqa"):
        positions = [index for index, value in enumerate(sources) if value == source]
        rng.shuffle(positions)
        selected.extend(positions[:16])
    pq.write_table(table.take(pa.array(selected)), DESTINATION)
    print(f"Saved {len(selected)} fixed prompts to {DESTINATION}")


if __name__ == "__main__":
    main()
