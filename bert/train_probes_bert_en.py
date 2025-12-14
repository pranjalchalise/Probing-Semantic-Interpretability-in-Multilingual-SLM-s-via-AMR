import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import json

# Same order as everywhere else
FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]


def load_data(embeddings_path: Path, labels_path: Path, features_path: Path = None):
    """
    Load embeddings + labels (+ optional metadata).
    embeddings: (N, L, H)
    labels:     (N, F)
    """
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    print(f"Loaded embeddings from {embeddings_path}: {embeddings.shape}")
    print(f"Loaded labels from {labels_path}:     {labels.shape}")

    assert embeddings.shape[0] == labels.shape[0], "N mismatch between embeddings and labels"
    assert labels.shape[1] == len(FEATURE_NAMES), "Feature count mismatch"

    metadata = None
    if features_path is not None and features_path.exists():
        metadata = pd.read_csv(features_path)

    return embeddings, labels, metadata


def train_probe(X_train, y_train, X_test, y_test, feature_name: str, layer_idx: int):
    """
    Train a single linear probe for one layer + one feature.
    """
    # If all labels are the same, we can't really learn anything.
    if len(np.unique(y_train)) < 2:
        return {
            "layer": layer_idx,
            "feature": feature_name,
            "accuracy": float(np.mean(y_test == y_test[0])),
            "f1": 0.0,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_train": int(np.sum(y_train)),
            "n_pos_test": int(np.sum(y_test)),
            "note": "constant_labels",
        }

    # Same setup as mBERT
    probe = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "layer": layer_idx,
        "feature": feature_name,
        "accuracy": float(acc),
        "f1": float(f1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": int(np.sum(y_train)),
        "n_pos_test": int(np.sum(y_test)),
    }


def train_all_probes(train_emb, train_labels, test_emb, test_labels):
    """
    Loop over all layers + all features and train a probe for each combo.
    """
    N_train, L, H = train_emb.shape
    N_test = test_emb.shape[0]
    F = train_labels.shape[1]

    # sanity checks
    assert test_emb.shape[1] == L
    assert test_emb.shape[2] == H
    assert test_labels.shape[1] == F

    print(f"\nTraining {L} layers × {F} features = {L * F} probes")
    print(f"Train examples: {N_train}, Test examples: {N_test}")

    results = []

    for layer_idx in range(L):
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"\nProcessing {layer_name}...")

        # slice out this layer: (N, H)
        X_train = train_emb[:, layer_idx, :]
        X_test = test_emb[:, layer_idx, :]

        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            y_train = train_labels[:, feat_idx]
            y_test = test_labels[:, feat_idx]

            res = train_probe(X_train, y_train, X_test, y_test, feat_name, layer_idx)
            res["layer_name"] = layer_name
            results.append(res)

            print(f"  {feat_name}: acc={res['accuracy']:.3f}, f1={res['f1']:.3f}")

    return results


def save_results(results, out_stem: Path):
    """
    Save results as both JSON (full) and CSV (table).
    out_stem: path without extension, e.g. data/bert_en_probe_results
    """
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_stem.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {json_path}")

    df = pd.DataFrame(results)
    csv_path = out_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    return df


def create_pivot(df_results: pd.DataFrame):
    """
    Build a layer × feature pivot of accuracy values.
    """
    pivot = df_results.pivot_table(
        index="layer_name",
        columns="feature",
        values="accuracy",
        aggfunc="mean",
    )

    # Put layers in a sensible order
    layer_order = ["embedding"] + [f"layer_{i}" for i in range(1, 13)]
    pivot = pivot.reindex([l for l in layer_order if l in pivot.index])
    return pivot


def print_summary(df_results: pd.DataFrame):
    """
    Print best layer per feature for quick inspection.
    """
    print("\n" + "=" * 80)
    print("BERT ENGLISH: Best layer per semantic feature")
    print("=" * 80)

    for feat in FEATURE_NAMES:
        feat_df = df_results[df_results["feature"] == feat]
        best = feat_df.loc[feat_df["accuracy"].idxmax()]

        print(f"\n{feat}:")
        print(f"  Best layer: {best['layer_name']} (layer {best['layer']})")
        print(f"  Accuracy:  {best['accuracy']:.3f}")
        print(f"  F1 score:  {best['f1']:.3f}")
        print(f"  Train examples: {best['n_train']}, Test examples: {best['n_test']}")
        print(f"  Positives (train/test): {best['n_pos_train']}/{best['n_pos_test']}")


def main():
    # Work relative to this file so paths aren't annoying
    this_dir = Path(__file__).resolve().parent
    data_dir = this_dir.parent / "data"
    bert_data_dir = data_dir / "bert"

    train_emb_path = bert_data_dir / "bert_en_train_cls_embeddings.npy"
    train_lbl_path = bert_data_dir / "bert_en_train_labels.npy"
    train_feat_path = bert_data_dir / "bert_en_train_features.csv"

    test_emb_path = bert_data_dir / "bert_en_test_cls_embeddings.npy"
    test_lbl_path = bert_data_dir / "bert_en_test_labels.npy"
    test_feat_path = bert_data_dir / "bert_en_test_features.csv"

    print("Loading BERT English train data...")
    train_emb, train_lbl, train_meta = load_data(
        train_emb_path, train_lbl_path, train_feat_path
    )

    print("\nLoading BERT English test data...")
    test_emb, test_lbl, test_meta = load_data(
        test_emb_path, test_lbl_path, test_feat_path
    )

    # Train everything
    results = train_all_probes(train_emb, train_lbl, test_emb, test_lbl)

    # Save
    out_stem = bert_data_dir / "probe_results"
    df_results = save_results(results, out_stem)

    print_summary(df_results)

    print("\n" + "=" * 80)
    print("BERT ENGLISH: Accuracy table (Layer × Feature)")
    print("=" * 80)
    pivot = create_pivot(df_results)
    print(pivot.round(3))

    pivot_path = bert_data_dir / "probe_results_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"\nSaved pivot table to {pivot_path}")


if __name__ == "__main__":
    main()
