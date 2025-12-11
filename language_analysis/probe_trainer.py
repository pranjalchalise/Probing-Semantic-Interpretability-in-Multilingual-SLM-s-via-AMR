# language_analysis/probe_trainer.py
#
# Functions to train probes for specific languages

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train_probe_for_language(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    language: str,
    layer_idx: int,
    feature_idx: int,
    feature_name: str
):
    """
    Train a single probe for a specific language, layer, and feature.
    
    Returns:
        dict with results
    """
    # Extract embeddings for this layer
    X_train = train_embeddings[:, layer_idx, :]
    X_test = test_embeddings[:, layer_idx, :]
    
    # Extract labels for this feature
    y_train = train_labels[:, feature_idx]
    y_test = test_labels[:, feature_idx]
    
    # Check if we have enough positive examples
    if len(np.unique(y_train)) < 2:
        return {
            "language": language,
            "layer": layer_idx,
            "layer_name": "embedding" if layer_idx == 0 else f"layer_{layer_idx}",
            "feature": feature_name,
            "accuracy": float(np.mean(y_test == y_test[0])),
            "f1": 0.0,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_pos_train": int(np.sum(y_train)),
            "n_pos_test": int(np.sum(y_test)),
            "note": "constant_labels"
        }
    
    # Train probe
    probe = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    
    probe.fit(X_train, y_train)
    y_pred = probe.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        "language": language,
        "layer": layer_idx,
        "layer_name": "embedding" if layer_idx == 0 else f"layer_{layer_idx}",
        "feature": feature_name,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_pos_train": int(np.sum(y_train)),
        "n_pos_test": int(np.sum(y_test)),
    }


def train_language_probes(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_metadata: pd.DataFrame,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_metadata: pd.DataFrame,
    languages: list,
    feature_names: list
):
    """
    Train probes for each language, layer, and feature combination.
    
    Returns:
        list of result dicts
    """
    N_layers = train_embeddings.shape[1]
    all_results = []
    
    for lang in languages:
        print(f"\nProcessing {lang}...")
        
        # Get indices for this language in train and test
        train_lang_mask = train_metadata['lang'] == lang
        test_lang_mask = test_metadata['lang'] == lang
        
        train_lang_indices = np.where(train_lang_mask)[0]
        test_lang_indices = np.where(test_lang_mask)[0]
        
        if len(train_lang_indices) == 0 or len(test_lang_indices) == 0:
            print(f"  Warning: No data for {lang} in train or test set. Skipping.")
            continue
        
        # Extract language-specific data
        lang_train_emb = train_embeddings[train_lang_indices]
        lang_train_labels = train_labels[train_lang_indices]
        lang_test_emb = test_embeddings[test_lang_indices]
        lang_test_labels = test_labels[test_lang_indices]
        
        print(f"  Train: {len(train_lang_indices)} examples, Test: {len(test_lang_indices)} examples")
        
        # Train probes for each layer and feature
        for layer_idx in range(N_layers):
            for feat_idx, feat_name in enumerate(feature_names):
                result = train_probe_for_language(
                    lang_train_emb, lang_train_labels,
                    lang_test_emb, lang_test_labels,
                    lang, layer_idx, feat_idx, feat_name
                )
                all_results.append(result)
        
        # Print summary for this language
        lang_results = [r for r in all_results if r['language'] == lang]
        if lang_results:
            best_by_feature = {}
            for feat in feature_names:
                feat_results = [r for r in lang_results if r['feature'] == feat]
                if feat_results:
                    best = max(feat_results, key=lambda x: x['accuracy'])
                    best_by_feature[feat] = best['layer_name']
            print(f"  Best layers: {best_by_feature}")
    
    return all_results

