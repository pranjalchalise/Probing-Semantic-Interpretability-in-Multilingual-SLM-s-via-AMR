# train_probes.py
#
# What this file does:
# - Load embeddings from each layer and semantic feature labels
# - Train linear probes (LogisticRegression) for each (layer, feature) combination
# - Evaluate probe accuracy and save results
# - Optionally create visualizations showing which layers encode which features best

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import json

# Feature names matching the order in the labels array
FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]


def load_data(embeddings_path, labels_path, features_path=None):
    """
    Load embeddings, labels, and optionally metadata (like language info).
    
    Args:
        embeddings_path: path to .npy file with shape (N, L, H)
                         N = num examples, L = num layers, H = hidden size
        labels_path: path to .npy file with shape (N, F)
                     N = num examples, F = num features
        features_path: optional path to CSV with metadata (e.g., language)
    
    Returns:
        embeddings: (N, L, H) array
        labels: (N, F) array
        metadata: DataFrame if features_path provided, else None
    """
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")
    
    # Verify shapes match
    assert embeddings.shape[0] == labels.shape[0], \
        f"Mismatch: {embeddings.shape[0]} examples in embeddings, {labels.shape[0]} in labels"
    assert labels.shape[1] == len(FEATURE_NAMES), \
        f"Expected {len(FEATURE_NAMES)} features, got {labels.shape[1]}"
    
    metadata = None
    if features_path and Path(features_path).exists():
        metadata = pd.read_csv(features_path)
        #assert len(metadata) == embeddings.shape[0], \
            #f"Metadata length {len(metadata)} doesn't match embeddings {embeddings.shape[0]}"
    
    return embeddings, labels, metadata


def train_probe(X_train, y_train, X_test, y_test, feature_name: str, layer_idx: int):
    """
    Train a single linear probe and evaluate it.
    
    Args:
        X_train: training embeddings for this layer, shape (N_train, H)
        y_train: training labels for this feature, shape (N_train,)
        X_test: test embeddings for this layer, shape (N_test, H)
        y_test: test labels for this feature, shape (N_test,)
        feature_name: name of the feature being predicted
        layer_idx: which layer this probe is for
    
    Returns:
        dict with accuracy, f1, and other metrics
    """
    # Check if we have enough positive examples
    if len(np.unique(y_train)) < 2:
        # All examples have the same label - can't train a meaningful probe
        return {
            "layer": layer_idx,
            "feature": feature_name,
            "accuracy": float(np.mean(y_test == y_test[0])),  # baseline accuracy
            "f1": 0.0,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_train": int(np.sum(y_train)),
            "n_pos_test": int(np.sum(y_test)),
            "note": "constant_labels"
        }
    
    # Train linear probe
    probe = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'  # good for small-medium datasets
    )
    
    probe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        "layer": layer_idx,
        "feature": feature_name,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": int(np.sum(y_train)),
        "n_pos_test": int(np.sum(y_test)),
    }


def train_all_probes(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    metadata: pd.DataFrame = None
):
    """
    Train probes for all (layer, feature) combinations using separate train/test splits.
    
    Args:
        train_embeddings: (N_train, L, H) array
        train_labels: (N_train, F) array
        test_embeddings: (N_test, L, H) array
        test_labels: (N_test, F) array
        metadata: optional DataFrame with metadata (for language-specific analysis)
    
    Returns:
        results: list of dicts, one per (layer, feature) combination
    """
    N_train, L, H = train_embeddings.shape
    N_test = test_embeddings.shape[0]
    F = train_labels.shape[1]
    
    # Verify shapes match
    assert train_embeddings.shape[1] == test_embeddings.shape[1], \
        "Train and test must have same number of layers"
    assert train_embeddings.shape[2] == test_embeddings.shape[2], \
        "Train and test must have same hidden size"
    assert train_labels.shape[1] == test_labels.shape[1], \
        "Train and test must have same number of features"
    
    print(f"\nTraining {L} layers × {F} features = {L * F} probes")
    print(f"Train examples: {N_train}, Test examples: {N_test}")
    
    results = []
    
    # Train a probe for each layer and each feature
    for layer_idx in range(L):
        layer_name = "embedding" if layer_idx == 0 else f"layer_{layer_idx}"
        print(f"\nProcessing {layer_name}...")
        
        # Extract embeddings for this layer: (N, H)
        train_layer_embeddings = train_embeddings[:, layer_idx, :]
        test_layer_embeddings = test_embeddings[:, layer_idx, :]
        
        for feat_idx, feature_name in enumerate(FEATURE_NAMES):
            # Extract labels for this feature: (N,)
            train_feature_labels = train_labels[:, feat_idx]
            test_feature_labels = test_labels[:, feat_idx]
            
            # Train probe
            result = train_probe(
                train_layer_embeddings, train_feature_labels,
                test_layer_embeddings, test_feature_labels,
                feature_name, layer_idx
            )
            result["layer_name"] = layer_name
            results.append(result)
            
            # Print progress
            print(f"  {feature_name}: accuracy={result['accuracy']:.3f}, f1={result['f1']:.3f}")
    
    return results


def save_results(results: list, output_path: str):
    """
    Save probe results to JSON and CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (full details)
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {json_path}")
    
    # Save as CSV (easier to analyze)
    df_results = pd.DataFrame(results)
    csv_path = output_path.with_suffix('.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")
    
    return df_results


def print_summary(df_results: pd.DataFrame):
    """
    Print a summary table showing best layer for each feature.
    """
    print("\n" + "="*80)
    print("SUMMARY: Best Layer for Each Semantic Feature")
    print("="*80)
    
    for feature in FEATURE_NAMES:
        feature_results = df_results[df_results['feature'] == feature]
        best = feature_results.loc[feature_results['accuracy'].idxmax()]
        
        print(f"\n{feature}:")
        print(f"  Best layer: {best['layer_name']} (layer {best['layer']})")
        print(f"  Accuracy: {best['accuracy']:.3f}")
        print(f"  F1 score: {best['f1']:.3f}")
        print(f"  Train examples: {best['n_train']}, Test examples: {best['n_test']}")
        print(f"  Positive examples (train/test): {best['n_pos_train']}/{best['n_pos_test']}")


def create_results_table(df_results: pd.DataFrame):
    """
    Create a pivot table showing accuracy for each (layer, feature) combination.
    """
    pivot = df_results.pivot_table(
        index='layer_name',
        columns='feature',
        values='accuracy',
        aggfunc='mean'
    )
    
    # Reorder layers logically
    layer_order = ['embedding'] + [f'layer_{i}' for i in range(1, 13)]
    pivot = pivot.reindex([l for l in layer_order if l in pivot.index])
    
    return pivot


def main():
    """
    Main function to run probe training.
    
    Uses separate train and test splits from the MASSIVE-AMR dataset.
    Make sure to run load_dataset.py and preprocess.py for both train and test sets first!
    """
    # Paths to train data files
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    mbert_data_dir = data_dir / "mbert"
    
    train_embeddings_path = mbert_data_dir / "mbert_train_cls_embeddings.npy"
    train_labels_path = data_dir / "massive_train_labels.npy"
    train_features_path = data_dir / "massive_train_features.csv"  # optional, for metadata
    
    # Paths to test data files
    test_embeddings_path = mbert_data_dir / "mbert_test_cls_embeddings.npy"
    test_labels_path = data_dir / "massive_test_labels.npy"
    test_features_path = data_dir / "massive_test_features.csv"  # optional, for metadata
    
    # Load train data
    print("Loading train data...")
    train_embeddings, train_labels, train_metadata = load_data(
        train_embeddings_path, train_labels_path, train_features_path
    )
    
    # Load test data
    print("Loading test data...")
    test_embeddings, test_labels, test_metadata = load_data(
        test_embeddings_path, test_labels_path, test_features_path
    )
    
    # Train all probes
    results = train_all_probes(
        train_embeddings, train_labels,
        test_embeddings, test_labels,
        metadata=train_metadata  # or test_metadata, or combine them
    )
    
    # Save results
    df_results = save_results(results, mbert_data_dir / "probe_results")
    
    # Print summary
    print_summary(df_results)
    
    # Create and display pivot table
    print("\n" + "="*80)
    print("ACCURACY TABLE: Layer × Feature")
    print("="*80)
    pivot = create_results_table(df_results)
    print(pivot.round(3))
    
    # Save pivot table
    pivot_path = mbert_data_dir / "probe_results_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"\nSaved pivot table to {pivot_path}")


if __name__ == "__main__":
    main()

