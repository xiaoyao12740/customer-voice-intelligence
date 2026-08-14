from __future__ import annotations

import html
import re
import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: str) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = URL_PATTERN.sub(" URL ", text)
    text = TAG_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def validate_text(value: str, max_length: int = 10000) -> str:
    text = clean_text(value)
    if not text:
        raise ValueError("Text must not be empty.")
    if len(text) > max_length:
        raise ValueError(f"Text exceeds the {max_length}-character limit.")
    return text


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"text", "sentiment", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["raw_text"] = result["text"].astype(str)
    result["clean_text"] = result["raw_text"].map(clean_text)
    before = len(result)
    empty = int((result["clean_text"] == "").sum())
    result = result[result["clean_text"] != ""].copy()
    duplicate_count = int(result.duplicated("clean_text").sum())
    result = result.drop_duplicates("clean_text").reset_index(drop=True)
    invalid = sorted(set(result["sentiment"]) - {"negative", "positive"})
    if invalid:
        raise ValueError(f"Invalid sentiment labels: {invalid}")
    report = {"rows_before": before, "empty_removed": empty, "duplicates_removed": duplicate_count, "rows_after": len(result)}
    return result, report

