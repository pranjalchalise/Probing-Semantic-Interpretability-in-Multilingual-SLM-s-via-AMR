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
    Create plots showing probe accuracy and F1 scores by layer for different languages.
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
    
    # Plot 2: F1 Score vs Layer for each language and feature
    for feat in feature_names:
        feat_results = results_df[results_df['feature'] == feat]
        
        # Get top languages by data size
        lang_counts = feat_results.groupby('language')['n_train'].first().sort_values(ascending=False)
        top_langs = lang_counts.head(10).index.tolist()  # Top 10 languages
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for lang in top_langs:
            lang_data = feat_results[feat_results['language'] == lang]
            lang_data = lang_data.sort_values('layer')
            
            # Filter out zero F1 scores (they indicate constant labels or poor performance)
            # But still plot them to show where probes fail
            ax.plot(lang_data['layer'], lang_data['f1'], 
                   marker='o', label=lang, linewidth=2, markersize=4)
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'{feat} Probe F1 Score by Layer (Top 10 Languages)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 13))
        ax.set_ylim(-0.05, 1.05)  # F1 is between 0 and 1
        
        plt.tight_layout()
        plt.savefig(output_dir / f'f1_by_layer_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved accuracy and F1 plots to {output_dir}")


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
    
    # Plot 3: F1 Score heatmap (best F1 per language and feature)
    best_f1_layers = []
    for lang in results_df['language'].unique():
        lang_results = results_df[results_df['language'] == lang]
        for feat in feature_names:
            feat_data = lang_results[lang_results['feature'] == feat]
            if len(feat_data) > 0:
                # Find layer with best F1 score (not accuracy)
                best_f1 = feat_data.loc[feat_data['f1'].idxmax()]
                best_f1_layers.append({
                    'language': lang,
                    'feature': feat,
                    'best_layer': best_f1['layer'],
                    'best_f1': best_f1['f1']
                })
    
    if best_f1_layers:
        best_f1_df = pd.DataFrame(best_f1_layers)
        f1_pivot = best_f1_df.pivot(index='language', columns='feature', values='best_f1')
        f1_pivot = f1_pivot.sort_index()
        
        plt.figure(figsize=(10, max(12, len(f1_pivot) * 0.3)))
        sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                    cbar_kws={'label': 'Best F1 Score'}, vmin=0, vmax=1)
        plt.title('Best F1 Score for Each Language and Feature', fontsize=14)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Language', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'best_f1_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Dual-axis plots showing both Accuracy and F1 for selected features
    # This is especially useful for imbalanced features
    for feat in feature_names:
        feat_results = results_df[results_df['feature'] == feat]
        
        # Get languages with sufficient test data (at least 50 test examples)
        lang_test_counts = feat_results.groupby('language')['n_test'].first()
        valid_langs = lang_test_counts[lang_test_counts >= 50].index.tolist()
        
        if len(valid_langs) == 0:
            continue
        
        # Get top 5 languages by data size
        lang_counts = feat_results[feat_results['language'].isin(valid_langs)].groupby('language')['n_train'].first().sort_values(ascending=False)
        top_langs = lang_counts.head(5).index.tolist()
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        for lang in top_langs:
            lang_data = feat_results[feat_results['language'] == lang]
            lang_data = lang_data.sort_values('layer')
            
            # Plot accuracy on left axis
            color = ax1.plot(lang_data['layer'], lang_data['accuracy'], 
                           marker='o', label=f'{lang} (Acc)', linewidth=2, markersize=4)[0].get_color()
            
            # Plot F1 on right axis with same color but dashed
            ax2 = ax1.twinx()
            ax2.plot(lang_data['layer'], lang_data['f1'], 
                    marker='s', label=f'{lang} (F1)', linewidth=2, 
                    markersize=4, linestyle='--', color=color, alpha=0.7)
        
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12, color='black')
        ax2.set_ylabel('F1 Score', fontsize=12, color='black')
        ax1.set_title(f'{feat} Probe: Accuracy and F1 Score by Layer (Top 5 Languages)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 13))
        ax1.set_ylim(0, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'accuracy_f1_dual_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved summary plots to {output_dir}")

