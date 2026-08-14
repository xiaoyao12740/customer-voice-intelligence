from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import load_config
from src.preprocessing import prepare_frame

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
SOURCE_FILES = {"amazon_cells_labelled.txt": "amazon", "imdb_labelled.txt": "imdb", "yelp_labelled.txt": "yelp"}


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent": "customer-voice-intelligence/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    config = load_config(); raw_dir = Path(config["data"]["raw_dir"]); processed = Path(config["data"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True); archive = PROJECT_ROOT / "data" / "sentiment-labelled-sentences.zip"
    if not archive.exists():
        print("Downloading the UCI Sentiment Labelled Sentences dataset..."); download(archive)
    if not raw_dir.exists():
        with ZipFile(archive) as bundle: bundle.extractall(PROJECT_ROOT / "data" / "raw")
    rows, malformed_lines = [], 0
    for filename, source in SOURCE_FILES.items():
        path = raw_dir / filename
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or "\t" not in line:
                malformed_lines += 1
                continue
            text, label = line.rsplit("\t", 1)
            if label not in {"0", "1"}:
                malformed_lines += 1
                continue
            rows.append({"text": text, "sentiment": "positive" if label == "1" else "negative", "source": source})
    frame, quality = prepare_frame(pd.DataFrame(rows))
    seed = int(config["project"]["seed"])
    train, remainder = train_test_split(frame, test_size=0.30, stratify=frame[["sentiment", "source"]].astype(str).agg("_".join, axis=1), random_state=seed)
    validation, test = train_test_split(remainder, test_size=0.50, stratify=remainder[["sentiment", "source"]].astype(str).agg("_".join, axis=1), random_state=seed)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.reset_index(drop=True).to_csv(processed / f"{name}.csv", index=False)
    quality["malformed_source_lines_skipped"] = malformed_lines
    summary = {"dataset": "UCI Sentiment Labelled Sentences", "doi": "10.24432/C57604", "license": "CC BY 4.0", "quality": quality, "splits": {"train": len(train), "validation": len(validation), "test": len(test)}}
    (processed / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
