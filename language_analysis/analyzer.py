# language_analysis/analyzer.py
#
# Analysis functions for language-specific probe results

import pandas as pd
import numpy as np


def is_english(lang_code: str) -> bool:
    """Check if language code is English."""
    return lang_code is not None and lang_code.startswith('en-')


def find_best_layer_per_language(results_df: pd.DataFrame, feature_names: list):
    """
    For each language and feature, find which layer has the highest accuracy.
    
    Returns:
        DataFrame with columns: language, feature, best_layer, best_accuracy
    """
    best_layers = []
    
    for lang in results_df['language'].unique():
        lang_results = results_df[results_df['language'] == lang]
        
        for feat in feature_names:
            feat_results = lang_results[lang_results['feature'] == feat]
            if len(feat_results) > 0:
                best = feat_results.loc[feat_results['accuracy'].idxmax()]
                best_layers.append({
                    'language': lang,
                    'feature': feat,
                    'best_layer': best['layer'],
                    'best_layer_name': best['layer_name'],
                    'best_accuracy': best['accuracy'],
                    'best_f1': best['f1']
                })
    
    return pd.DataFrame(best_layers)


def analyze_results(results_df: pd.DataFrame, feature_names: list, output_dir=None):
    """
    Main analysis function. Returns summary statistics.
    
    Args:
        results_df: DataFrame with probe results
        feature_names: list of feature names
        output_dir: optional Path for saving results (if None, uses default)
    """
    from pathlib import Path
    
    best_layers_df = find_best_layer_per_language(results_df, feature_names)
    
    print("\nBest Layer per Language and Feature:")
    print("-" * 80)
    for lang in sorted(best_layers_df['language'].unique()):
        lang_best = best_layers_df[best_layers_df['language'] == lang]
        print(f"\n{lang}:")
        for _, row in lang_best.iterrows():
            print(f"  {row['feature']}: {row['best_layer_name']} (layer {row['best_layer']}) "
                  f"- Accuracy: {row['best_accuracy']:.3f}")
    
    # Save best layers
    if output_dir is None:
        output_path = Path("./data/best_layers_by_language.csv")
    else:
        output_dir = Path(output_dir)
        output_path = output_dir / "best_layers_by_language.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_layers_df.to_csv(output_path, index=False)
    print(f"\nSaved best layers to {output_path}")
    
    return {
        'best_layers': best_layers_df,
        'results_df': results_df
    }


def compare_languages(results_df: pd.DataFrame, feature_names: list, output_dir=None):
    """
    Compare English vs non-English languages.
    Tests hypothesis: Non-English languages encode semantic info in later layers.
    
    Args:
        results_df: DataFrame with probe results
        feature_names: list of feature names
        output_dir: optional Path for saving results (if None, uses default)
    """
    from pathlib import Path
    
    best_layers_df = find_best_layer_per_language(results_df, feature_names)
    
    # Add English flag
    best_layers_df['is_english'] = best_layers_df['language'].apply(is_english)
    
    print("\nEnglish vs Non-English Comparison:")
    print("-" * 80)
    
    comparison_results = {}
    
    for feat in feature_names:
        feat_data = best_layers_df[best_layers_df['feature'] == feat]
        
        english_data = feat_data[feat_data['is_english'] == True]
        non_english_data = feat_data[feat_data['is_english'] == False]
        
        if len(english_data) > 0 and len(non_english_data) > 0:
            eng_avg_layer = english_data['best_layer'].mean()
            non_eng_avg_layer = non_english_data['best_layer'].mean()
            
            eng_avg_acc = english_data['best_accuracy'].mean()
            non_eng_avg_acc = non_english_data['best_accuracy'].mean()
            
            print(f"\n{feat}:")
            print(f"  English: avg layer {eng_avg_layer:.1f}, avg accuracy {eng_avg_acc:.3f}")
            print(f"  Non-English: avg layer {non_eng_avg_layer:.1f}, avg accuracy {non_eng_avg_acc:.3f}")
            print(f"  Layer difference: {non_eng_avg_layer - eng_avg_layer:.1f} "
                  f"({'later' if non_eng_avg_layer > eng_avg_layer else 'earlier'} for non-English)")
            
            comparison_results[feat] = {
                'english_avg_layer': eng_avg_layer,
                'non_english_avg_layer': non_eng_avg_layer,
                'layer_difference': non_eng_avg_layer - eng_avg_layer,
                'english_avg_accuracy': eng_avg_acc,
                'non_english_avg_accuracy': non_eng_avg_acc,
                'accuracy_difference': non_eng_avg_acc - eng_avg_acc
            }
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_results).T
    
    if output_dir is None:
        output_path = Path("./data/english_vs_nonenglish_comparison.csv")
    else:
        output_dir = Path(output_dir)
        output_path = output_dir / "english_vs_nonenglish_comparison.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path)
    print(f"\nSaved comparison to {output_path}")
    
    return comparison_df

