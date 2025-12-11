# xlmr/train_probes_xlmr.py
#
# Train linear probes on XLM-R CLS embeddings (all languages).
# Same setup as mBERT/BERT:
#   - LogisticRegression, lbfgs, max_iter=1000
#   - Features: ARG0, ARG1, ARG2, neg, time
#   - Train on MASSIVE train, eval on MASSIVE test.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import json

FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]


def load_data(embeddings_path: Path, labels_path: Path, features_path: Path = None):
    emb = np.load(embeddings_path)
    lab = np.load(labels_path)

    print(f"Loaded embeddings: {embeddings_path} {emb.shape}")
    print(f"Loaded labels:     {labels_path} {lab.shape}")

    assert emb.shape[0] == lab.shape[0], "N mismatch between embeddings and labels"
    assert lab.shape[1] == len(FEATURE_NAMES), "Feature count mismatch"

    meta = None
    if features_path is not None and features_path.exists():
        meta = pd.read_csv(features_path)

    return emb, lab, meta


def train_probe(X_train, y_train, X_test, y_test, feature_name: str, layer_idx: int):
    # Constant labels = useless probe
    if len(np.unique(y_train)) < 2:
        return {
            "layer": layer_idx,
            "layer_name": "embedding" if layer_idx == 0 else f"layer_{layer_idx}",
            "feature": feature_name,
            "accuracy": float(np.mean(y_test == y_test[0])),
            "f1": 0.0,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_train": int(np.sum(y_train)),
            "n_pos_test": int(np.sum(y_test)),
            "note": "constant_labels",
        }

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
        "layer_name": "embedding" if layer_idx == 0 else f"layer_{layer_idx}",
        "feature": feature_name,
        "accuracy": float(acc),
        "f1": float(f1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": int(np.sum(y_train)),
        "n_pos_test": int(np.sum(y_test)),
    }


def train_all_probes(train_emb, train_lab, test_emb, test_lab):
    N_train, L, H = train_emb.shape
    N_test = test_emb.shape[0]
    F = train_lab.shape[1]

    assert test_emb.shape[1] == L
    assert test_emb.shape[2] == H
    assert test_lab.shape[1] == F

    print(f"\nTraining {L} layers × {F} features = {L * F} probes")
    print(f"Train examples: {N_train}, Test examples: {N_test}")

    results = []

    for layer_idx in range(L):
        print(f"\nLayer {layer_idx} ({'embedding' if layer_idx == 0 else f'layer_{layer_idx}'})")
        X_train = train_emb[:, layer_idx, :]
        X_test = test_emb[:, layer_idx, :]

        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            y_train = train_lab[:, feat_idx]
            y_test = test_lab[:, feat_idx]

            res = train_probe(X_train, y_train, X_test, y_test, feat_name, layer_idx)
            results.append(res)

            print(f"  {feat_name}: acc={res['accuracy']:.3f}, f1={res['f1']:.3f}")

    return results


def save_results(results, out_stem: Path):
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
    pivot = df_results.pivot_table(
        index="layer_name",
        columns="feature",
        values="accuracy",
        aggfunc="mean",
    )
    order = ["embedding"] + [f"layer_{i}" for i in range(1, 13)]
    pivot = pivot.reindex([l for l in order if l in pivot.index])
    return pivot


def print_summary(df_results: pd.DataFrame):
    print("\n" + "=" * 80)
    print("XLM-R (all languages): Best layer per feature")
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
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    train_emb_path = data_dir / "xlmr_train_cls_embeddings.npy"
    train_lbl_path = data_dir / "xlmr_train_labels.npy"
    train_feat_path = data_dir / "massive_train_features.csv"

    test_emb_path = data_dir / "xlmr_test_cls_embeddings.npy"
    test_lbl_path = data_dir / "xlmr_test_labels.npy"
    test_feat_path = data_dir / "massive_test_features.csv"

    print("Loading XLM-R train data...")
    train_emb, train_lab, train_meta = load_data(
        train_emb_path, train_lbl_path, train_feat_path
    )

    print("\nLoading XLM-R test data...")
    test_emb, test_lab, test_meta = load_data(
        test_emb_path, test_lbl_path, test_feat_path
    )

    results = train_all_probes(train_emb, train_lab, test_emb, test_lab)

    df_results = save_results(results, data_dir / "xlmr_probe_results")

    print_summary(df_results)

    print("\n" + "=" * 80)
    print("XLM-R: Accuracy table (Layer × Feature)")
    print("=" * 80)
    pivot = create_pivot(df_results)
    print(pivot.round(3))

    pivot.to_csv(data_dir / "xlmr_probe_results_pivot.csv")
    print(f"\nSaved pivot table to {data_dir / 'xlmr_probe_results_pivot.csv'}")


if __name__ == "__main__":
    main()
