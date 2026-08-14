import pytest
import pandas as pd
from src.preprocessing import clean_text, prepare_frame, validate_text


def test_clean_text_preserves_meaning():
    assert clean_text("<b>Great!</b>  https://example.com") == "Great! URL"


def test_validate_rejects_empty_and_long():
    with pytest.raises(ValueError): validate_text("   ")
    with pytest.raises(ValueError): validate_text("x" * 10001)


def test_prepare_frame_removes_duplicates():
    frame=pd.DataFrame([{"text":"Good","sentiment":"positive","source":"demo"},{"text":"Good","sentiment":"positive","source":"demo"}])
    result,report=prepare_frame(frame); assert len(result)==1 and report["duplicates_removed"]==1

