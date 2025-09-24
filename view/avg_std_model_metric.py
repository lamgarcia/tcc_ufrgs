import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult.csv")
df = pd.read_csv(csv_path)


metric = "precision"

metric = "accuracy"
metric = "disparate_impact"
metric = "equalized_odds_difference"
metric = "average_predictive_value_difference"

# Cria identificador de pipeline
def get_pipeline(row):
    techniques = []
    if row["pre"] != "none":
        techniques.append("pre")
    if row["in"] != "none":
        techniques.append("in")
    if row["post"] != "none":
        techniques.append("post")
    return "+".join(techniques) if techniques else "baseline"

df["pipeline"] = df.apply(get_pipeline, axis=1)

# Ordem fixa para os pipelines
pipeline_order = ["baseline", "pre", "in", "post",
                  "pre+in", "pre+post", "in+post", "pre+in+post"]

# Agrupa por modelo e pipeline → média e desvio
summary = (
    df.groupby(["model", "pipeline"])[metric]
      .agg(["mean", "std"])
      .reset_index()
)

# Reordena pipelines
summary["pipeline"] = pd.Categorical(summary["pipeline"], categories=pipeline_order, ordered=True)
summary = summary.sort_values(["model", "pipeline"])

# === Plot ===
plt.figure(figsize=(10,6))

for model, group in summary.groupby("model"):
    plt.errorbar(
        group["pipeline"],
        group["mean"],
        yerr=group["std"],
        fmt="-o",
        capsize=5,
        markersize=8,
        linewidth=2,
        label=model
    )

plt.xlabel("Mitigação")
plt.ylabel(f"Média ± Desvio de {metric}")
plt.title(f"{metric} e modelos")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Modelo")
plt.tight_layout()
plt.show()
