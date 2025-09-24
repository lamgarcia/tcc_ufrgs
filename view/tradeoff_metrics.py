import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult.csv")
df = pd.read_csv(csv_path)

# Dicionário de marcadores por modelo
markers = ["o", "s", "^", "D", "v", "<", ">", "*","."]
marker_map = {model: markers[i % len(markers)] for i, model in enumerate(df["model"].unique())}

plt.figure(figsize=(10,6))

for model, group in df.groupby("model"):
    plt.scatter(
        group["disparate_impact"],
        group["equal_opportunity_ratio"],
        c=group["accuracy"],
        cmap="viridis",
        s=120,
        alpha=0.9,
        marker=marker_map[model],
        label=model
    )

# Linhas de referência
plt.axvline(1.0, color="black", linestyle=":", label="DI = 1 (equidade perfeita)")
plt.axhline(1.0, color="red", linestyle=":", label="APVD = 1 (equidade perfeita)")

plt.xlabel("Disparate Impact (quanto mais próximo de 1, melhor)")
plt.ylabel("equal_opportunity_ratio (quanto mais próximo de 0, melhor)")
plt.title("Trade-off: DI vs equal_opportunity_ratio (cores = Accuracy, formas = Modelos)")
plt.colorbar(label="Accuracy")
plt.legend(title="Modelo")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
