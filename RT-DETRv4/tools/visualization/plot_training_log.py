#!/usr/bin/env python3
"""
Plot loss and AP curves from RT-DETRv4 training logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: Path):
    loss_by_epoch = OrderedDict()
    ap_by_epoch = OrderedDict()
    current_epoch = None

    epoch_re = re.compile(r"Epoch:\s*\[(\d+)\]")
    loss_re = re.compile(r"loss:\s*([0-9.]+)\s*\(([0-9.]+)\)")
    ap_re = re.compile(r"Average Precision\s+\(AP\)\s+\@\[\s*IoU=0.50:0.95.*\]\s*=\s*([0-9.]+)")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except Exception:
                payload = None

            if isinstance(payload, dict):
                epoch_val = payload.get("epoch")
                if epoch_val is not None:
                    current_epoch = int(epoch_val)
                    if "train_loss" in payload:
                        loss_by_epoch[current_epoch] = float(payload["train_loss"])
                    elif "loss" in payload:
                        loss_by_epoch[current_epoch] = float(payload["loss"])

                    test_bbox = payload.get("test_coco_eval_bbox")
                    if isinstance(test_bbox, (list, tuple)) and test_bbox:
                        ap_by_epoch[current_epoch] = float(test_bbox[0])
                continue

            epoch_match = epoch_re.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                loss_match = loss_re.search(line)
                if loss_match:
                    loss_by_epoch[current_epoch] = float(loss_match.group(2))

            ap_match = ap_re.search(line)
            if ap_match and current_epoch is not None:
                ap_by_epoch[current_epoch] = float(ap_match.group(1))

    return loss_by_epoch, ap_by_epoch


def parse_events(summary_dir: Path):
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:  # pragma: no cover
        print(f"[plot] TensorBoard not available: {exc}")
        return {}, {}

    summary_dir = summary_dir.resolve()
    accumulator = event_accumulator.EventAccumulator(
        str(summary_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])

    loss_by_step = OrderedDict()
    ap_by_epoch = OrderedDict()

    if "Loss/total" in tags:
        for item in accumulator.Scalars("Loss/total"):
            loss_by_step[item.step] = item.value

    if "Test/coco_eval_bbox_0" in tags:
        for item in accumulator.Scalars("Test/coco_eval_bbox_0"):
            ap_by_epoch[item.step] = item.value

    return loss_by_step, ap_by_epoch


def save_plot(data, title, ylabel, out_path: Path, xlabel: str = "Epoch"):
    if not data:
        print(f"[plot] No data for {title}, skip.")
        return
    epochs = list(data.keys())
    values = list(data.values())
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, values, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[plot] Saved {out_path}")


def save_csv(rows, header, out_path: Path):
    if not rows:
        print(f"[plot] No data for {out_path.name}, skip.")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[plot] Saved {out_path}")


def smooth_series(values, window: int):
    if window <= 1 or len(values) < window:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, help="Path to log.txt")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNGs")
    parser.add_argument("--summary-dir", type=Path, default=None, help="TensorBoard summary dir")
    parser.add_argument("--smooth", type=int, default=10, help="Moving average window size")
    args = parser.parse_args()

    log_path = args.log
    out_dir = args.out_dir or log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_by_epoch, ap_by_epoch = parse_log(log_path)
    used_events = False

    if (not loss_by_epoch and not ap_by_epoch) or args.summary_dir:
        summary_dir = args.summary_dir or (log_path.parent / "summary")
        if summary_dir.exists():
            loss_by_epoch, ap_by_epoch = parse_events(summary_dir)
            used_events = True

    loss_vals = list(loss_by_epoch.values())
    loss_vals = smooth_series(loss_vals, args.smooth)
    loss_by_epoch = OrderedDict(zip(loss_by_epoch.keys(), loss_vals))

    ap_vals = list(ap_by_epoch.values())
    ap_vals = smooth_series(ap_vals, args.smooth)
    ap_by_epoch = OrderedDict(zip(ap_by_epoch.keys(), ap_vals))

    save_plot(
        loss_by_epoch,
        title="Training Loss (avg per epoch)",
        ylabel="Loss",
        out_path=out_dir / "loss_curve.png",
        xlabel="Step" if used_events else "Epoch",
    )
    save_plot(
        ap_by_epoch,
        title="Validation AP (IoU=0.50:0.95)",
        ylabel="AP",
        out_path=out_dir / "ap_curve.png",
    )

    axis_label = "step" if used_events else "epoch"
    loss_rows = [(k, v) for k, v in loss_by_epoch.items()]
    ap_rows = [(k, v) for k, v in ap_by_epoch.items()]
    save_csv(loss_rows, [axis_label, "loss"], out_dir / "loss_epoch.csv")
    save_csv(ap_rows, [axis_label, "ap"], out_dir / "ap_epoch.csv")

    all_keys = sorted(set(loss_by_epoch.keys()) | set(ap_by_epoch.keys()))
    metrics_rows = [
        (k, loss_by_epoch.get(k, ""), ap_by_epoch.get(k, ""))
        for k in all_keys
    ]
    save_csv(metrics_rows, [axis_label, "loss", "ap"], out_dir / "metrics_epoch.csv")


if __name__ == "__main__":
    main()

