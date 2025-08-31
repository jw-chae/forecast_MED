
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import translators as ts

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')
UNIFIED_FILE = os.path.join(PROCESSED_DIR, 'final_unified_dataset.json')
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

# Set global font for matplotlib to handle Chinese characters
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

# --- Helper Functions ---
def load_unified_data(file_path):
    """Loads the unified JSON dataset into a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def translate_for_plots(text):
    """A simple translation wrapper for plot labels and titles."""
    if not text or not isinstance(text, str):
        return ""
    try:
        if text not in translate_for_plots.cache:
            translate_for_plots.cache[text] = ts.translate_text(text, translator='google', to_language='en')
        return translate_for_plots.cache[text]
    except Exception as e:
        print(f"Could not translate '{text}': {e}")
        return text
translate_for_plots.cache = {}

def save_plot(fig, filename):
    """Saves a matplotlib figure to the visualizations directory."""
    if not os.path.exists(VISUALIZATIONS_DIR):
        os.makedirs(VISUALIZATIONS_DIR)
    path = os.path.join(VISUALIZATIONS_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filename}")

# --- Advanced Visualization ---
def plot_diagnosis_distribution(df, column='diagnosis', threshold=0.01):
    """
    Plots the distribution of diagnoses, grouping rare ones into 'Other'.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Skipping plot.")
        return

    # Drop rows where diagnosis is missing
    df = df.dropna(subset=[column])
    
    # Calculate frequencies
    counts = df[column].value_counts()
    total = len(df[column])
    frequencies = counts / total
    
    # Identify rare diagnoses
    rare_mask = frequencies < threshold
    rare_diagnoses = frequencies[rare_mask].index
    
    # Group rare diagnoses into 'Other'
    df_plot = df.copy()
    df_plot[column] = df_plot[column].replace(rare_diagnoses, 'Other')
    
    # Recalculate counts for the new grouped data
    plot_counts = df_plot[column].value_counts()

    # Translate index for plotting
    translated_labels = [translate_for_plots(label) for label in plot_counts.index]

    # Create the plot
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=plot_counts.values, y=translated_labels, palette='plasma')
    
    ax.set_title('Distribution of Primary Diagnoses (Rare Diagnoses Grouped)', fontsize=16)
    ax.set_xlabel('Number of Cases', fontsize=12)
    ax.set_ylabel('Diagnosis', fontsize=12)
    
    # Add percentage labels to the bars
    for i, v in enumerate(plot_counts.values):
        ax.text(v + 1, i, f' {v} ({plot_counts.values[i]/total*100:.1f}%)', color='black', va='center')

    plt.tight_layout()
    save_plot(plt.gcf(), 'advanced_diagnosis_distribution.png')


def main():
    """Main function to load data and generate advanced visualizations."""
    print("Loading unified data for visualization...")
    df = load_unified_data(UNIFIED_FILE)
    
    # We will use the 'diagnosis' column from the unified file.
    # This corresponds to '主诊断' from the source files.
    plot_diagnosis_distribution(df, column='primary_diagnosis', threshold=0.01)
    
    print("Advanced visualization script finished.")

if __name__ == "__main__":
    main()
