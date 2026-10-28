#!/usr/bin/env python3
"""Import an offline veRL/W&B run into a SwanLab experiment.

This reads history only; it never starts training. W&B files are read through
wandb's internal protobuf reader because the run is offline and may not exist
on the W&B server.
"""
import argparse, csv, json, os
from pathlib import Path
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2


def read_history(path):
    store = datastore.DataStore()
    store.open_for_scan(str(path))
    rows = []
    while True:
        blob = store.scan_data()
        if blob is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(blob)
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                row[key] = json.loads(item.value_json)
            except Exception:
                continue
        if row:
            step = row.get("training/global_step", row.get("_step"))
            if step is not None:
                row["_step"] = int(float(step))
                rows.append(row)
    # Same step can occur in separate records; merge all fields.
    merged = {}
    for row in rows:
        merged.setdefault(row["_step"], {}).update(row)
    return [merged[k] for k in sorted(merged)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wandb_file", type=Path)
    ap.add_argument("--out", type=Path, default=Path("logs/swanlab/searchr1-ppo-import"))
    ap.add_argument("--final-em", type=float)
    ap.add_argument("--final-step", type=int)
    ap.add_argument("--project", default="Search-R1")
    ap.add_argument("--experiment", default="ppo-qwen2.5-3b-imported")
    ap.add_argument("--mode", choices=("local", "offline"), default="local")
    args = ap.parse_args()
    rows = read_history(args.wandb_file)
    if not rows:
        raise SystemExit("No history records found")
    args.out.mkdir(parents=True, exist_ok=True)
    # Only add an external final EM when both values are explicitly supplied.
    if (args.final_em is None) != (args.final_step is None):
        ap.error("--final-em and --final-step must be supplied together")
    if args.final_em is not None:
        final = next((r for r in rows if r["_step"] == args.final_step), None)
        if final is None:
            final = {"_step": args.final_step}
            rows.append(final)
        final["val-core/nq/reward/mean@1"] = args.final_em
        rows.sort(key=lambda r: r["_step"])
    keys = sorted({k for r in rows for k in r})
    with (args.out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    (args.out / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    import swanlab
    run = swanlab.init(project=args.project, experiment_name=args.experiment,
                       description="Imported Search-R1 veRL W&B history; no training executed by this import.",
                       config={"source": str(args.wandb_file), "final_step": args.final_step},
                       logdir=str(args.out), mode=args.mode)
    for row in rows:
        metrics = {k: v for k, v in row.items() if k != "_step" and isinstance(v, (int, float))}
        if metrics:
            run.log(metrics, step=row["_step"])
    run.finish()
    print(f"Imported {len(rows)} steps into {args.out}")
    print(f"CSV: {args.out / 'metrics.csv'}")

if __name__ == "__main__":
    main()
