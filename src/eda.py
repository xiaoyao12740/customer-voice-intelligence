from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run_eda(frame: pd.DataFrame, output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    lengths = frame["clean_text"].str.split().map(len)
    summary = {
        "rows": len(frame), "missing": frame.isna().sum().to_dict(),
        "duplicate_clean_text": int(frame.duplicated("clean_text").sum()),
        "sentiment_counts": frame["sentiment"].value_counts().to_dict(),
        "source_counts": frame["source"].value_counts().to_dict(),
        "text_length_words": {"mean": float(lengths.mean()), "median": float(lengths.median()), "p95": float(lengths.quantile(.95)), "max": int(lengths.max())},
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    frame["sentiment"].value_counts().sort_index().plot.bar(ax=axes[0], color=["#d9534f", "#2ca58d"], title="Sentiment")
    frame["source"].value_counts().sort_index().plot.bar(ax=axes[1], color="#3478bf", title="Source")
    axes[2].hist(lengths, bins=30, color="#7e57c2"); axes[2].set_title("Text length (words)")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(output / "dataset_distribution.png", dpi=180); plt.close(fig)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

