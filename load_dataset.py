# Parse each block into (sentence, AMR graph, language)
# Extract simple semantic features from the AMR (ARG0, ARG1, ARG2, neg, time)
# Return / save everything as a pandas DataFrame

import penman
import pandas as pd
from pathlib import Path

# These are the AMR-based features we’re currently using.
FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]


def parse_lang_from_id(id_line: str):
    """
    Example line: "# ::id 13973-ur-PK"
    I just want to pull out "ur-PK" from that.
    """
    if not id_line:
        return None

    # Strip the "# ::id" prefix so we get "13973-ur-PK"
    id_val = id_line.replace('# ::id', '').strip()

    # Split once on the first '-' so that we get ["13973", "ur-PK"]
    parts = id_val.split('-', 1)
    if len(parts) == 2:
        return parts[1]
    return None


def load_amr_text(path: str):
    """
    Read a MASSIVE-AMR .txt file and turn it into a list of dicts:
      {"sentence": ..., "graph": ..., "lang": ...}

    - sentence: the natural language sentence (from ::snt)
    - graph: parsed AMR graph (penman.Graph), or None if parsing fails
    - lang: language code like "en-US", "ur-PK", etc. (from ::id)
    """
    data = []
    path = Path(path)

    # Try UTF-8 first, but handle encoding errors gracefully
    try:
        with path.open('r', encoding='utf-8', errors='replace') as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        # Fallback: try latin-1 which can decode any byte
        with path.open('r', encoding='latin-1', errors='replace') as f:
            content = f.read().strip()

    # Each entry is one example, separated by a blank line
    entries = content.split("\n\n")

    for entry in entries:
        # Split into non-empty lines
        lines = [l for l in entry.strip().split("\n") if l.strip()]
        if not lines:
            continue

        # Sentence line looks like: "# ::snt some sentence here"
        snt_line = next((l for l in lines if l.startswith('# ::snt')), None)
        sentence = snt_line.replace('# ::snt', '').strip() if snt_line else None

        # ID line looks like "# ::id 13973-ur-PK"
        id_line = next((l for l in lines if l.startswith('# ::id')), None)
        lang = parse_lang_from_id(id_line)

        # AMR is whatever is *not* a comment line
        amr_str = "\n".join(l for l in lines if not l.startswith('#')).strip()

        graph = None
        if amr_str:
            try:
                graph = penman.decode(amr_str)
            except Exception:
                # If it fails to parse, just keep graph=None and move on
                graph = None

        data.append({"sentence": sentence, "graph": graph, "lang": lang})

    return data


def extract_features(graph):
    """
    Take a penman.Graph and turn it into a small set of binary features:
      - ARG0 / ARG1 / ARG2 present?
      - negation present?
      - any time-related role present? (time/day/month/year)
    """
    if graph is None:
        # If we couldn't parse the graph, just say "no" for everything
        return {feat: 0 for feat in FEATURE_NAMES}

    # Each edge is (source, role, target). I only care about the role names.
    roles = [role.lstrip(":") for (_, role, _) in graph.edges()]

    return {
        "ARG0": int("ARG0" in roles),
        "ARG1": int("ARG1" in roles),
        "ARG2": int("ARG2" in roles),
        "neg": int("polarity" in roles),
        "time": int(any(r in roles for r in ["time", "day", "month", "year"])),
    }


def build_dataframe(amr_path: str) -> pd.DataFrame:
    """
    Full pipeline for one file:
      - read AMR text
      - parse graphs
      - extract features
      - return a DataFrame with columns:
          sentence, lang, ARG0, ARG1, ARG2, neg, time
    """
    dataset = load_amr_text(amr_path)

    rows = []
    for item in dataset:
        feats = extract_features(item["graph"])
        rows.append({
            "sentence": item["sentence"],
            "lang": item["lang"],
            **feats
        })

    df = pd.DataFrame(rows)
    return df


def save_dataframe(df: pd.DataFrame, out_path: str, fmt: str = "parquet"):
    """
    Generic saver. We tell it the format with `fmt`:
      - "table"
      - "csv"
      - "jsonl" (one JSON object per line)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = fmt.lower()
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "jsonl":
        with out_path.open('w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(row.to_json(force_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_dataframe_json(df: pd.DataFrame, out_path: str):
    """
    Convenience wrapper just for saving as JSONL, so you can call
    save_dataframe(...) for parquet and save_dataframe_json(...) for json
    without thinking about the `fmt` argument.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(row.to_json(force_ascii=False) + "\n")


if __name__ == "__main__":
    amr_path_val = "./data/amrs-massive-val.txt"
    amr_path_train = "./data/amrs-massive-train.txt"  # for train
    amr_path_test = "./data/amrs-massive-test.txt"  # for test

    #df_val = build_dataframe(amr_path_val)
    #print(df_val.sample(10).to_string())

    # Save as CSV (no extra dependencies needed)
    #save_dataframe(df_val, "./data/massive_val_features.csv", fmt="csv")

    # And also save as JSONL (one JSON object per line)
    #save_dataframe_json(df, "./data/massive_val_features.jsonl")

    df_train = build_dataframe(amr_path_train)
    #print(df_train.sample(10).to_string())

    # Save as CSV (no extra dependencies needed)
    save_dataframe(df_train, "./data/massive_train_features.csv", fmt="csv")

    # And also save as JSONL (one JSON object per line)
    save_dataframe_json(df_train, "./data/massive_triain_features.jsonl")


    df_test = build_dataframe(amr_path_test)
    print(df_test.sample(10).to_string())

    # Save as CSV (no extra dependencies needed)
    save_dataframe(df_test, "./data/massive_test_features.csv", fmt="csv")

    # And also save as JSONL (one JSON object per line)
    save_dataframe_json(df_test, "./data/massive_test_features.jsonl")