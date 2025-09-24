import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult.csv")
df = pd.read_csv(csv_path)

model = "xgboost"
label_performance = "accuracy"
label_discrimination = "disparate_impact"

df = df[df["model"].str.lower() == model].copy()

# Cria identificador de pipeline (baseado nas técnicas aplicadas)
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

# Define marcador de acordo com o pipeline
marker_map = {
    "baseline": "x",
    "pre": "o",
    "in": "s",
    "post": "^",
    "pre+in": "D",
    "pre+post": "v",
    "in+post": "<",
    "pre+in+post": "*",
}

df["marker"] = df["pipeline"].map(marker_map)

# === Plot ===
plt.figure(figsize=(10,6))

for _, row in df.iterrows():
    plt.scatter(
        row[label_discrimination],   # eixo X = fairness
        row[label_performance],      # eixo Y = performance
        marker=row["marker"], 
        s=120, alpha=0.8, edgecolor="black"
    )

plt.xlabel(label_discrimination)
plt.ylabel(label_performance)
plt.title(label_performance + " vs " + label_discrimination +" (" + model+")")
plt.grid(True, linestyle="--", alpha=0.5)

# Linha de perfeito balanço (DI = 1)
plt.axvline(x=1.0, color="red", linestyle="--", linewidth=1)

# Legenda manual
legend_elements = [
    mlines.Line2D([], [], color="black", marker=marker, linestyle="None", markersize=10, label=label)
    for label, marker in marker_map.items()
]
plt.legend(handles=legend_elements, title="Mitigação", loc="best")

plt.tight_layout()
plt.show()
