import pandas as pd
import os

def prepare_hk_data():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data_for_model", "手足口病", "data_HK", "hk_hfmd_weekly_2010_2025.csv")
    output_dir = os.path.join(base_dir, "data_for_model", "手足口病", "data_HK")
    output_path = os.path.join(output_dir, "hk_hfmd_baseline_ready.csv")

    # Read data
    df = pd.read_csv(input_path)
    
    # Select and rename columns
    # We need 'date' and 'value'
    df_ready = df[['week_end_date', 'admissions_total']].copy()
    df_ready.rename(columns={'week_end_date': 'date', 'admissions_total': 'value'}, inplace=True)
    
    # Ensure date is datetime
    df_ready['date'] = pd.to_datetime(df_ready['date'])
    
    # Sort by date
    df_ready.sort_values('date', inplace=True)
    
    # Save
    df_ready.to_csv(output_path, index=False)
    print(f"Created {output_path}")
    print(df_ready.head())

if __name__ == "__main__":
    prepare_hk_data()
