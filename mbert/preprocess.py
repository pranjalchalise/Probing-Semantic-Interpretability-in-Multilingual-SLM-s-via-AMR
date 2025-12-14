# preprocess.py
#
# What this file does:
# - Load a transformer model (e.g. mBERT) with hidden states turned on
# - Take a list of sentences and pull out the CLS vector from every layer
# - Save those CLS embeddings + the labels so we can train probes later

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# If there's a GPU, use it. Otherwise, just stay on CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = "bert-base-multilingual-cased"):
    """
    Load a HuggingFace encoder model and its tokenizer.

    By default I’m using mBERT:
        "bert-base-multilingual-cased"

    But I can swap this string out for:
        - "xlm-roberta-base"
        - "bert-base-uncased"
        - etc.

    I set output_hidden_states=True so that the model returns all layer outputs,
    not just the last layer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True
    ).to(device)

    # I'm not fine-tuning the model here, just reading representations.
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
    Given a list of sentences, return the CLS embedding from every layer.

    Args:
        sentences: list of strings (one sentence per example)
        tokenizer, model: loaded from load_model()
        batch_size: how many sentences to process in one forward pass
        max_length: maximum number of tokens per sentence
                    (anything longer gets truncated)
        as_numpy:
            - True  -> return a NumPy array
            - False -> return a torch.Tensor

    Returns:
        embeddings with shape (N, L, H)
          N = number of sentences
          L = number of layers + 1 (embedding layer + transformer layers)
          H = hidden size (e.g. 768 for BERT-base)
    """
    all_batches = []

    # Go through the sentences in small chunks so we don't blow up memory.
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        # Tokenize this batch and send it to the right device
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
            # hidden_states is a tuple with one entry per layer:
            # len(hidden_states) = num_layers + 1 (embeddings + each transformer block)
            # each entry has shape (batch_size, seq_len, hidden_size)

        # For each layer, grab the CLS token (index 0 along the sequence dimension)
        # That gives a list of tensors, each of shape (batch_size, hidden_size)
        layer_cls = [h[:, 0, :] for h in hidden_states]

        # Stack these along a new "layer" dimension:
        # result: (batch_size, L, H)
        batch_emb = torch.stack(layer_cls, dim=1)

        # Move to CPU so GPU memory doesn't slowly fill up across batches
        all_batches.append(batch_emb.cpu())

    # Combine all batches into one big tensor: (N, L, H)
    all_emb = torch.cat(all_batches, dim=0)

    if as_numpy:
        return all_emb.numpy()
    return all_emb


def save_numpy(arr: np.ndarray, out_path: str):
    """
    Save a NumPy array as a .npy file.

    Using .npy instead of JSON because:
    - it’s smaller on disk
    - much faster to load/save
    - and it keeps the exact dtypes/shapes without extra work
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def example_run():
    """
    Quick end-to-end run to check that everything is wired up.

    Assumes load_dataset.py has already been run and produced:
        ./data/massive_val_features.csv

    That CSV should at least have the columns:
        sentence, lang, ARG0, ARG1, ARG2, neg, time
    """

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    mbert_data_dir = data_dir / "mbert"
    mbert_data_dir.mkdir(parents=True, exist_ok=True)
    
    #Did this for all three datasets separately
    feature_path = data_dir / "massive_test_features.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Could not find {feature_path}. "
            "Run load_dataset.py first so the features CSV exists."
        )

    # Load the sentence-level AMR features from CSV
    df = pd.read_csv(feature_path)

    # Just in case, drop any rows where sentence is missing
    df = df.dropna(subset=["sentence"])

    # If I ever want to focus on a single language, I can filter here.
    # For example, to keep only English rows (if lang looks like "en-US"):
    # df = df[df["lang"].str.startswith("en")]

    sentences = df["sentence"].tolist()

    # Load the encoder I want to probe (starting with mBERT)
    tokenizer, model = load_model("bert-base-multilingual-cased")

    # Get CLS embeddings from every layer for all sentences
    embeddings = get_cls_embeddings(
        sentences,
        tokenizer,
        model,
        batch_size=16,
        max_length=64,
        as_numpy=True,
    )
    print("Embeddings shape:", embeddings.shape)  # should be (N, L, H)

    # Save CLS embeddings for later use in probe training
    save_numpy(embeddings, mbert_data_dir / "mbert_test_cls_embeddings.npy")

    # Save the AMR-based labels (ARG0, ARG1, ARG2, neg, time) as a separate array
    label_arr = df[["ARG0", "ARG1", "ARG2", "neg", "time"]].to_numpy()
    save_numpy(label_arr, data_dir / "massive_test_labels.npy")


if __name__ == "__main__":
    example_run()
