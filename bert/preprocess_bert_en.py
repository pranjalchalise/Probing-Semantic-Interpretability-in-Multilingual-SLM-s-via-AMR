# bert/preprocess_bert_en.py
#
# Use BERT-base (English-only) to get CLS embeddings for English sentences.
# Input:
#   ../data/bert_en_train_features.csv
#   ../data/bert_en_test_features.csv
# Output:
#   ../data/bert_en_train_cls_embeddings.npy
#   ../data/bert_en_test_cls_embeddings.npy
#   ../data/bert_en_train_labels.npy
#   ../data/bert_en_test_labels.npy

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = "bert-base-uncased"):
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

        # CLS token from every layer
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
      ../data/bert/bert_en_<split>_features.csv
    Produces:
      ../data/bert/bert_en_<split>_cls_embeddings.npy
      ../data/bert/bert_en_<split>_labels.npy
    """
    this_dir = Path(__file__).resolve().parent
    data_dir = this_dir.parent / "data"
    bert_data_dir = data_dir / "bert"
    bert_data_dir.mkdir(parents=True, exist_ok=True)

    feature_path = bert_data_dir / f"bert_en_{split}_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Could not find {feature_path}")

    df = pd.read_csv(feature_path).dropna(subset=["sentence"])
    sentences = df["sentence"].tolist()

    tokenizer, model = load_model("bert-base-uncased")

    embeddings = get_cls_embeddings(
        sentences,
        tokenizer,
        model,
        batch_size=16,
        max_length=64,
        as_numpy=True,
    )
    print(f"[{split}] Embeddings shape:", embeddings.shape)

    save_numpy(embeddings, bert_data_dir / f"bert_en_{split}_cls_embeddings.npy")

    label_arr = df[["ARG0", "ARG1", "ARG2", "neg", "time"]].to_numpy()
    save_numpy(label_arr, bert_data_dir / f"bert_en_{split}_labels.npy")


if __name__ == "__main__":
    run_for_split("train")
    run_for_split("test")
