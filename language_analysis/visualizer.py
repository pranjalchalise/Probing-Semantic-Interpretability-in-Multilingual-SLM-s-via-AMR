# language_analysis/visualizer.py
#
# Visualization functions for language-specific analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def plot_results(results_df: pd.DataFrame, feature_names: list, output_dir: Path):
    """
    Create plots showing probe accuracy by layer for different languages.
    """
    output_dir = Path(output_dir)
    
    # Plot 1: Accuracy vs Layer for each language and feature
    for feat in feature_names:
        feat_results = results_df[results_df['feature'] == feat]
        
        # Get top languages by data size
        lang_counts = feat_results.groupby('language')['n_train'].first().sort_values(ascending=False)
        top_langs = lang_counts.head(10).index.tolist()  # Top 10 languages
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lang in top_langs:
            lang_data = feat_results[feat_results['language'] == lang]
            lang_data = lang_data.sort_values('layer')
            
            ax.plot(lang_data['layer'], lang_data['accuracy'], 
                   marker='o', label=lang, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{feat} Probe Accuracy by Layer (Top 10 Languages)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 13))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'accuracy_by_layer_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved accuracy plots to {output_dir}")


def create_summary_plots(results_df: pd.DataFrame, comparison_df: pd.DataFrame, 
                        feature_names: list, output_dir: Path):
    """
    Create summary plots comparing English vs non-English.
    """
    output_dir = Path(output_dir)
    
    # Import here to avoid circular imports
    from language_analysis.analyzer import find_best_layer_per_language
    
    # Plot 1: Best layer comparison (English vs Non-English)
    fig, axes = plt.subplots(1, len(feature_names), figsize=(5*len(feature_names), 6))
    if len(feature_names) == 1:
        axes = [axes]
    
    for idx, feat in enumerate(feature_names):
        feat_data = results_df[results_df['feature'] == feat]
        best_layers = []
        
        for lang in feat_data['language'].unique():
            lang_data = feat_data[feat_data['language'] == lang]
            best = lang_data.loc[lang_data['accuracy'].idxmax()]
            best_layers.append({
                'language': lang,
                'best_layer': best['layer'],
                'is_english': lang.startswith('en-')
            })
        
        best_df = pd.DataFrame(best_layers)
        
        # Box plot
        english_layers = best_df[best_df['is_english']]['best_layer'].values
        non_english_layers = best_df[~best_df['is_english']]['best_layer'].values
        
        if len(english_layers) > 0 and len(non_english_layers) > 0:
            axes[idx].boxplot([english_layers, non_english_layers], 
                             labels=['English', 'Non-English'])
            axes[idx].set_ylabel('Best Layer', fontsize=12)
            axes[idx].set_title(f'{feat}', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Best Layer Comparison: English vs Non-English', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_layer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Heatmap of best layers by language and feature
    best_layers_df = find_best_layer_per_language(results_df, feature_names)
    
    pivot = best_layers_df.pivot(index='language', columns='feature', values='best_layer')
    pivot = pivot.sort_index()
    
    plt.figure(figsize=(10, max(12, len(pivot) * 0.3)))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis', 
                cbar_kws={'label': 'Best Layer'})
    plt.title('Best Layer for Each Language and Feature', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Language', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'best_layer_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plots to {output_dir}")

