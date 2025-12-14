# analyze_by_language_xlmr.py
#
# Main script to analyze XLM-R probe performance by language
# Tests hypothesis: Non-English languages encode semantic roles in later layers
# Uses XLM-RoBERTa embeddings instead of mBERT

import numpy as np
import pandas as pd
from pathlib import Path
from language_analysis.probe_trainer import train_language_probes
from language_analysis.analyzer import analyze_results, compare_languages
from language_analysis.visualizer import plot_results, create_summary_plots

FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]
MIN_EXAMPLES_PER_LANG = 100  # Minimum examples needed to train probes


def load_data_with_languages(embeddings_path: str, labels_path: str, features_path: str):
    """
    Load embeddings, labels, and metadata with language information.
    Ensures all arrays are aligned.
    """
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    metadata = pd.read_csv(features_path)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Loaded metadata: {len(metadata)} rows")
    
    # Ensure alignment - filter metadata to match embeddings
    n_embeddings = embeddings.shape[0]
    if len(metadata) > n_embeddings:
        print(f"Warning: Metadata has {len(metadata)} rows, embeddings have {n_embeddings}. Truncating metadata.")
        metadata = metadata.iloc[:n_embeddings].copy()
    elif len(metadata) < n_embeddings:
        print(f"Warning: Metadata has {len(metadata)} rows, embeddings have {n_embeddings}.")
        # This shouldn't happen, but handle it
        raise ValueError("Metadata has fewer rows than embeddings. Check data alignment.")
    
    # Verify shapes
    assert embeddings.shape[0] == labels.shape[0] == len(metadata), \
        f"Shape mismatch: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}, metadata={len(metadata)}"
    
    return embeddings, labels, metadata


def filter_languages_by_data_size(metadata: pd.DataFrame, min_examples: int = MIN_EXAMPLES_PER_LANG):
    """
    Filter languages that have enough data for meaningful probe training.
    Returns list of language codes and their counts.
    """
    lang_counts = metadata['lang'].value_counts()
    valid_langs = lang_counts[lang_counts >= min_examples].index.tolist()
    
    print(f"\nLanguages with at least {min_examples} examples:")
    for lang in sorted(valid_langs):
        print(f"  {lang}: {lang_counts[lang]} examples")
    
    return valid_langs, lang_counts


def main():
    """
    Main analysis pipeline for XLM-R:
    1. Load data with language information
    2. Filter languages with sufficient data
    3. Train probes for each language
    4. Analyze results
    5. Generate visualizations
    """
    print("="*80)
    print("XLM-R Language-Specific Probe Analysis")
    print("="*80)
    
    # Paths to XLM-R data files
    train_embeddings_path = "./data/xlmr_train_cls_embeddings.npy"
    train_labels_path = "./data/xlmr_train_labels.npy"
    train_features_path = "./data/massive_train_features.csv"
    
    test_embeddings_path = "./data/xlmr_test_cls_embeddings.npy"
    test_labels_path = "./data/xlmr_test_labels.npy"
    test_features_path = "./data/massive_test_features.csv"
    
    # Load train data
    print("\nLoading XLM-R train data...")
    train_embeddings, train_labels, train_metadata = load_data_with_languages(
        train_embeddings_path, train_labels_path, train_features_path
    )
    
    # Load test data
    print("\nLoading XLM-R test data...")
    test_embeddings, test_labels, test_metadata = load_data_with_languages(
        test_embeddings_path, test_labels_path, test_features_path
    )
    
    # Filter languages with sufficient data
    print("\nFiltering languages by data size...")
    valid_langs, lang_counts = filter_languages_by_data_size(
        train_metadata, min_examples=MIN_EXAMPLES_PER_LANG
    )
    
    if len(valid_langs) == 0:
        print("No languages have sufficient data. Exiting.")
        return
    
    # Train probes for each language
    print(f"\nTraining XLM-R probes for {len(valid_langs)} languages...")
    all_results = train_language_probes(
        train_embeddings, train_labels, train_metadata,
        test_embeddings, test_labels, test_metadata,
        valid_langs, FEATURE_NAMES
    )
    
    # Save raw results
    results_df = pd.DataFrame(all_results)
    results_path = "./data/xlmr_language_probe_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")
    
    # Analyze results
    print("\n" + "="*80)
    print("XLM-R ANALYSIS")
    print("="*80)
    analysis_results = analyze_results(results_df, FEATURE_NAMES)
    
    # Save best layers with XLM-R prefix (note: analyze_results also saves to default path)
    best_layers_df = analysis_results['best_layers']
    best_layers_df.to_csv("./data/xlmr_best_layers_by_language.csv", index=False)
    print(f"Saved XLM-R best layers to ./data/xlmr_best_layers_by_language.csv")
    
    # Compare English vs non-English
    print("\n" + "="*80)
    print("XLM-R: ENGLISH vs NON-ENGLISH COMPARISON")
    print("="*80)
    comparison = compare_languages(results_df, FEATURE_NAMES)
    
    # Save comparison with XLM-R prefix (note: compare_languages also saves to default path)
    comparison.to_csv("./data/xlmr_english_vs_nonenglish_comparison.csv")
    print(f"Saved XLM-R comparison to ./data/xlmr_english_vs_nonenglish_comparison.csv")
    
    # Note: The analyzer functions save to default paths which will contain XLM-R results
    # The xlmr_ prefixed files above are the ones to use for XLM-R analysis
    
    # Generate visualizations
    print("\nGenerating XLM-R visualizations...")
    output_dir = Path("./data/xlmr_language_analysis_plots")
    output_dir.mkdir(exist_ok=True)
    
    plot_results(results_df, FEATURE_NAMES, output_dir)
    create_summary_plots(results_df, comparison, FEATURE_NAMES, output_dir)
    
    print(f"\nVisualizations saved to {output_dir}")
    print("\nXLM-R analysis complete!")


if __name__ == "__main__":
    main()

