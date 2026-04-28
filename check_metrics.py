import os
import pandas as pd

runs_dir = 'c:/Users/archit/Desktop/pbl/weapon_detection/runs/detect'
best_model = None
highest_map = -1

print("Comparing all trained models...\n")

for model_folder in os.listdir(runs_dir):
    csv_path = os.path.join(runs_dir, model_folder, 'results.csv')
    if os.path.exists(csv_path):
        try:
            # Read CSV and strip whitespace from column names just in case
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            # Find the best mAP50-95 score across all epochs for this model
            if 'metrics/mAP50-95(B)' in df.columns:
                max_map = df['metrics/mAP50-95(B)'].max()
                print(f"Model: {model_folder:<25} | Best mAP50-95: {max_map:.4f}")
                
                if max_map > highest_map:
                    highest_map = max_map
                    best_model = model_folder
        except Exception as e:
            print(f"Could not read {model_folder}: {e}")

print("\n--- CONCLUSION ---")
if best_model:
    print(f"The best performing model is '{best_model}' with a peak accuracy score of {highest_map:.4f}!")
else:
    print("Could not find any valid results to compare.")
