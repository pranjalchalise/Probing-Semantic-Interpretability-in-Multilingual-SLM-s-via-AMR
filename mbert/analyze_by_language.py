import numpy as np
import pandas as pd
from pathlib import Path
from language_analysis.probe_trainer import train_language_probes
from language_analysis.analyzer import analyze_results, compare_languages
from language_analysis.visualizer import plot_results, create_summary_plots

FEATURE_NAMES = ["ARG0", "ARG1", "ARG2", "neg", "time"]
MIN_EXAMPLES_PER_LANG = 100  # min examples to train probes


def load_data_with_languages(embeddings_path, labels_path, features_path):
    """
    loading embeddings, labels, and metadata with language information.
    """

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    metadata = pd.read_csv(features_path)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Loaded metadata: {len(metadata)} rows")
    
    # filtering to match embeddings
    n_embeddings = embeddings.shape[0]
    if len(metadata) > n_embeddings:
        print(f"Warning: Metadata has {len(metadata)} rows, embeddings have {n_embeddings}")
        metadata = metadata.iloc[:n_embeddings].copy()
    elif len(metadata) < n_embeddings:
        print(f"Warning: Metadata has {len(metadata)} rows, embeddings have {n_embeddings}.")
    
    return embeddings, labels, metadata


def filter_languages_by_data_size(metadata: pd.DataFrame, min_examples: int = MIN_EXAMPLES_PER_LANG):
    """
    Filter languages that have enough data for meaningful probe training
    """

    lang_counts = metadata['lang'].value_counts()
    valid_langs = lang_counts[lang_counts >= min_examples].index.tolist()
    
    print(f"\nLanguages with at least {min_examples} examples:")
    for lang in sorted(valid_langs):
        print(f"  {lang}: {lang_counts[lang]} examples")
    
    return valid_langs, lang_counts


def main():
    """
    Main analysis pipeline:
    1. Load data with language information
    2. Filter languages with sufficient data
    3. Train probes for each language
    4. Analyze results
    5. Generate visualizations
    """
    print("="*80)
    print("Language-Specific Probe Analysis")
    print("="*80)
    
    
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    mbert_data_dir = data_dir / "mbert"
    
    train_embeddings_path = mbert_data_dir / "mbert_train_cls_embeddings.npy"
    train_labels_path = data_dir / "massive_train_labels.npy"
    train_features_path = data_dir / "massive_train_features.csv"
    
    test_embeddings_path = mbert_data_dir / "mbert_test_cls_embeddings.npy"
    test_labels_path = data_dir / "massive_test_labels.npy"
    test_features_path = data_dir / "massive_test_features.csv"

    print("\nLoading train data...")
    train_embeddings, train_labels, train_metadata = load_data_with_languages(
        train_embeddings_path, train_labels_path, train_features_path
    )
    
    print("\nLoading test data...")
    test_embeddings, test_labels, test_metadata = load_data_with_languages(
        test_embeddings_path, test_labels_path, test_features_path
    )
    
    # filter per language w/ sufficient data
    print("\nFiltering languages by data size...")
    valid_langs, lang_counts = filter_languages_by_data_size(
        train_metadata, min_examples=MIN_EXAMPLES_PER_LANG
    )
    
    if len(valid_langs) == 0:
        print("No languages have sufficient data. Exiting.")
        return
    
    # training probes for each language
    print(f"\nTraining probes for {len(valid_langs)} languages...")
    all_results = train_language_probes(
        train_embeddings, train_labels, train_metadata,
        test_embeddings, test_labels, test_metadata,
        valid_langs, FEATURE_NAMES
    )
    
    results_df = pd.DataFrame(all_results)
    results_path = mbert_data_dir / "language_probe_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to {results_path}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    analysis_results = analyze_results(results_df, FEATURE_NAMES, output_dir=mbert_data_dir)
    
    #comparing english vs non englishs
    print("\n" + "="*80)
    print("ENGLISH vs NON-ENGLISH COMPARISON")
    print("="*80)
    comparison = compare_languages(results_df, FEATURE_NAMES, output_dir=mbert_data_dir)
    
    print("\nGenerating visualizations...")
    output_dir = mbert_data_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_results(results_df, FEATURE_NAMES, output_dir)
    create_summary_plots(results_df, comparison, FEATURE_NAMES, output_dir)
    
    print(f"\nVisualizations saved to {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

