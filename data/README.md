# Dataset / 数据集

The project uses the UCI Sentiment Labelled Sentences dataset: 3,000 English review sentences from Amazon, IMDb, and Yelp, balanced between positive and negative sentiment.

项目使用 UCI Sentiment Labelled Sentences：来自 Amazon、IMDb 和 Yelp 的 3,000 条英文评论，正负情感平衡。

- DOI: https://doi.org/10.24432/C57604
- License: CC BY 4.0
- Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

Run `python scripts/prepare_data.py` to download and create reproducible 70/15/15 stratified splits. Raw and processed data are intentionally excluded from Git.

