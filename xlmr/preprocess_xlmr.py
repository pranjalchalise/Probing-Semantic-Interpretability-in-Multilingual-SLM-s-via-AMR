# xlmr/preprocess_xlmr.py
#
# Run XLM-R (xlm-roberta-base) on MASSIVE train/test sentences (all languages),

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = "xlm-roberta-base"):
    """
    Load XLM-R + tokenizer, with hidden states turned on.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).to(device)
    model.eval()
    return tokenizer, model


def get_cls_embeddings(
    sentences,
    tokenizer,
    model,
    batch_size: int = 16,
    max_length: int = 64,
    as_numpy: bool = True,
):
    """
    Same idea as before:
    - tokenize sentences
    - run through model
    - take the first token (index 0) from every layer
    - stack into shape (N, L, H)
    """
    all_batches = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            hidden_states = outputs.hidden_states

        # For XLM-R, position 0 is the <s> token; we use it like CLS
        layer_cls = [h[:, 0, :] for h in hidden_states]
        batch_emb = torch.stack(layer_cls, dim=1)
        all_batches.append(batch_emb.cpu())

    all_emb = torch.cat(all_batches, dim=0)
    return all_emb.numpy() if as_numpy else all_emb


def save_numpy(arr: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def run_for_split(split: str):
    """
    split: "train" or "test"

    Expects:
      ../data/massive_<split>_features.csv
    Produces:
      ../data/xlmr/xlmr_<split>_cls_embeddings.npy
      ../data/xlmr/xlmr_<split>_labels.npy
    """
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    xlmr_data_dir = data_dir / "xlmr"
    xlmr_data_dir.mkdir(parents=True, exist_ok=True)

    feature_path = data_dir / f"massive_{split}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Could not find {feature_path}")

    df = pd.read_csv(feature_path).dropna(subset=["sentence"])
    sentences = df["sentence"].tolist()

    print(f"[{split}] Loaded {len(sentences)} sentences")

    tokenizer, model = load_model("xlm-roberta-base")

    embeddings = get_cls_embeddings(
        sentences,
        tokenizer,
        model,
        batch_size=16,
        max_length=64,
        as_numpy=True,
    )
    print(f"[{split}] XLM-R embeddings shape:", embeddings.shape)

    save_numpy(embeddings, xlmr_data_dir / f"xlmr_{split}_cls_embeddings.npy")

    # labels are the same AMR features as before
    label_arr = df[["ARG0", "ARG1", "ARG2", "neg", "time"]].to_numpy()
    save_numpy(label_arr, xlmr_data_dir / f"xlmr_{split}_labels.npy")


if __name__ == "__main__":
    run_for_split("train")
    run_for_split("test")
