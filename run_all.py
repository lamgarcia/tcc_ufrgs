import os
import pandas as pd
from train import run_experiment

CONFIG_DIR = "config"

def main():
    results = []

    for file in os.listdir(CONFIG_DIR):
        if file.endswith(".yaml"):
            config_path = os.path.join(CONFIG_DIR, file)
            print(f"Running experiment with {config_path} ...")
            result = run_experiment(config_path)
            results.append(result)

    df_results = pd.concat(results, ignore_index=True)
    print("\n=== Final Results ===")
    print(df_results)

    df_results.to_csv("runs.csv", index=False)

if __name__ == "__main__":
    main()
