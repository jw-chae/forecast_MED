
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import translators as ts
import squarify

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

# --- Detailed Diagnosis Visualization ---
def plot_top_n_diagnoses(df, column='primary_diagnosis', n=20):
    """
    Plots a horizontal bar chart of the top N most frequent diagnoses.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Skipping plot.")
        return

    df = df.dropna(subset=[column])
    top_n = df[column].value_counts().nlargest(n)
    
    translated_labels = [translate_for_plots(label) for label in top_n.index]

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=top_n.values, y=translated_labels, palette='mako')
    
    ax.set_title(f'Top {n} Primary Diagnoses', fontsize=16)
    ax.set_xlabel('Number of Cases', fontsize=12)
    ax.set_ylabel('Diagnosis', fontsize=12)
    
    for i, v in enumerate(top_n.values):
        ax.text(v + 0.5, i, str(v), color='black', va='center')

    plt.tight_layout()
    save_plot(plt.gcf(), f'top_{n}_diagnoses_dist.png')

def plot_diagnosis_treemap(df, column='primary_diagnosis', n=20):
    """
    Plots a treemap of the top N most frequent diagnoses.
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Skipping plot.")
        return

    df = df.dropna(subset=[column])
    top_n = df[column].value_counts().nlargest(n)
    
    translated_labels = [f"{translate_for_plots(label)}\n({count})" for label, count in top_n.items()]

    plt.figure(figsize=(16, 10))
    colors = sns.color_palette("viridis", len(top_n))
    
    squarify.plot(sizes=top_n.values, label=translated_labels, color=colors, alpha=0.8, text_kwargs={'fontsize':10, 'fontname':'Noto Sans CJK JP'})
    
    plt.title(f'Treemap of Top {n} Primary Diagnoses', fontsize=18)
    plt.axis('off')
    
    save_plot(plt.gcf(), f'top_{n}_diagnoses_treemap.png')

def plot_diagnosis_by_gender(df, diagnosis_col='primary_diagnosis', gender_col='gender', n=20):
    """
    Plots a stacked bar chart of gender distribution for the top N diagnoses.
    """
    if diagnosis_col not in df.columns or gender_col not in df.columns:
        print(f"Required columns not found. Skipping gender distribution plot.")
        return

    df_filtered = df.dropna(subset=[diagnosis_col, gender_col])
    top_diagnoses = df_filtered[diagnosis_col].value_counts().nlargest(n).index
    df_top = df_filtered[df_filtered[diagnosis_col].isin(top_diagnoses)]

    # Create a contingency table
    contingency_table = pd.crosstab(df_top[diagnosis_col], df_top[gender_col])
    contingency_table = contingency_table.loc[top_diagnoses] # Keep original sort order

    translated_labels = [translate_for_plots(label) for label in contingency_table.index]
    
    ax = contingency_table.plot(kind='barh', stacked=True, figsize=(14, 10), colormap='viridis')
    
    ax.set_title(f'Gender Distribution for Top {n} Diagnoses', fontsize=16)
    ax.set_xlabel('Number of Cases', fontsize=12)
    ax.set_ylabel('Diagnosis', fontsize=12)
    ax.set_yticklabels(translated_labels)
    
    plt.tight_layout()
    save_plot(plt.gcf(), f'top_{n}_diagnoses_by_gender.png')

def plot_diagnosis_by_age(df, diagnosis_col='primary_diagnosis', age_col='age', n=10):
    """
    Plots box plots of age distribution for the top N diagnoses.
    """
    if diagnosis_col not in df.columns or age_col not in df.columns:
        print(f"Required columns not found. Skipping age distribution plot.")
        return

    df_filtered = df.dropna(subset=[diagnosis_col, age_col])
    # Ensure age is numeric
    df_filtered[age_col] = pd.to_numeric(df_filtered[age_col], errors='coerce')
    df_filtered = df_filtered.dropna(subset=[age_col])

    top_diagnoses = df_filtered[diagnosis_col].value_counts().nlargest(n).index
    df_top = df_filtered[df_filtered[diagnosis_col].isin(top_diagnoses)]

    translated_labels = [translate_for_plots(label) for label in top_diagnoses]

    plt.figure(figsize=(14, 8))
    sns.boxplot(y=df_top[diagnosis_col], x=df_top[age_col], order=top_diagnoses, palette='viridis')
    
    plt.title(f'Age Distribution for Top {n} Diagnoses', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Diagnosis', fontsize=12)
    plt.yticks(ticks=range(len(translated_labels)), labels=translated_labels)
    
    plt.tight_layout()
    save_plot(plt.gcf(), f'top_{n}_diagnoses_by_age.png')

def main():
    """Main function to load data and generate detailed diagnosis visualizations."""
    print("Loading unified data for visualization...")
    df = load_unified_data(UNIFIED_FILE)
    
    plot_top_n_diagnoses(df, column='primary_diagnosis', n=20)
    plot_diagnosis_treemap(df, column='primary_diagnosis', n=20)
    plot_diagnosis_by_gender(df, diagnosis_col='primary_diagnosis', gender_col='gender', n=20)
    plot_diagnosis_by_age(df, diagnosis_col='primary_diagnosis', age_col='age', n=10)
    
    print("Detailed diagnosis visualization script finished.")

if __name__ == "__main__":
    main()
