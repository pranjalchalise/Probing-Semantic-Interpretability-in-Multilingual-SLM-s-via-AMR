# bert/make_english_splits.py
#
# Take the full MASSIVE-* features CSVs and make English-only versions
# for the BERT baseline.

import pandas as pd
from pathlib import Path


def filter_english(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path)

    # keep only rows where lang starts with "en-"
    en_df = df[df["lang"].fillna("").str.startswith("en-")].copy()

    print(f"{in_path}: {len(df)} total rows, {len(en_df)} English rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    en_df.to_csv(out_path, index=False)
    print(f"Saved English-only split to {out_path}")


if __name__ == "__main__":
    this_dir = Path(__file__).resolve().parent

    data_dir = this_dir.parent / "data"

    train_in = data_dir / "massive_train_features.csv"
    test_in  = data_dir / "massive_test_features.csv"

    filter_english(train_in, data_dir / "bert_en_train_features.csv")
    filter_english(test_in,  data_dir / "bert_en_test_features.csv")
