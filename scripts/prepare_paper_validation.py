"""Create a small, deterministic seven-dataset validation set for training checks."""

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    table = pq.read_table(args.data_dir / "test.parquet")
    sources = table.column("data_source").to_pylist()
    rng = random.Random(42)
    selected = []
    for source in sorted(set(sources)):
        positions = [i for i, value in enumerate(sources) if value == source]
        rng.shuffle(positions)
        chosen = positions[: args.per_source]
        selected.extend(chosen)
        print(f"{source}: {len(chosen)} / {len(positions)}")

    destination = args.output or args.data_dir / "validation_500_each.parquet"
    pq.write_table(table.take(pa.array(selected)), destination)
    print(f"Saved {len(selected)} examples to {destination}")


if __name__ == "__main__":
    main()
