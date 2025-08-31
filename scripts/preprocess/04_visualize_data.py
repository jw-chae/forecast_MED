
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import translators as ts

# Define paths
BASE_DIR = '/home/joongwon00/Project_Tsinghua_Paper/med_deepseek'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')
UNIFIED_FILE = os.path.join(PROCESSED_DIR, 'unified_dataset.json')
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' # Found font path

# Set global font for matplotlib to handle Chinese characters
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False  # Fix for minus sign not showing up

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
        # Using a simple cache to avoid re-translating the same text
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

# --- Visualization Functions ---
def plot_numerical_distributions(df):
    """Generates histograms and box plots for numerical columns."""
    sns.set_theme(style="whitegrid")
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].nunique() > 1: # Only plot if there's more than one value
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
            axes[0].set_title(f'Histogram of {col}')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Frequency')
            
            # Box Plot
            sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
            axes[1].set_title(f'Box Plot of {col}')
            axes[1].set_xlabel(col)
            
            plt.suptitle(f'Distribution of {col}', fontsize=16)
            save_plot(fig, f'numerical_dist_{col}.png')

def plot_categorical_distributions(df):
    """Generates bar charts for key categorical columns."""
    sns.set_theme(style="whitegrid")
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 50]
    for col in categorical_cols:
        if 'name' in col or 'id' in col: continue # Skip name and id columns
        
        plt.figure(figsize=(12, 8))
        ax = sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
        
        title = translate_for_plots(f'Distribution of {col}')
        xlabel = translate_for_plots('Count')
        ylabel = translate_for_plots(col)
        
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        save_plot(plt.gcf(), f'categorical_dist_{col}.png')

def generate_word_clouds(df):
    """Generates word clouds for text-heavy columns."""
    text_cols = ['chief_complaint', 'diagnosis', 'imaging_findings', 'content']
    for col in text_cols:
        if col in df.columns:
            # Combine all text, handling potential None values
            text = ' '.join(df[col].dropna().astype(str))
            if text:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    font_path=FONT_PATH
                ).generate(text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                title = translate_for_plots(f'Word Cloud for {col}')
                plt.title(title)
                save_plot(plt.gcf(), f'wordcloud_{col}.png')

def main():
    """Main function to load data and generate all visualizations."""
    print("Loading unified data for visualization...")
    df = load_unified_data(UNIFIED_FILE)
    
    print("Generating numerical data visualizations...")
    plot_numerical_distributions(df)
    
    print("Generating categorical data visualizations...")
    plot_categorical_distributions(df)
    
    print("Generating word clouds...")
    generate_word_clouds(df)
    
    print("Visualization script finished.")

if __name__ == "__main__":
    main()
