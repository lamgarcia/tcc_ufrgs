#inspirado nos graficos de : Kamiran, Data processing techiniques...

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Lê o CSV ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult_new_1.csv")
df = pd.read_csv(csv_path)

model = "xgboost"
label_performance = "f1-score"
label_discrimination = "disparate_impact"

df = df[df["model"].str.lower() == model].copy()

# Cria identificador de pipeline
df["pipeline"] = df["pre"] + "_" + df["in"] + "_" + df["post"]

# Conta quantas técnicas foram aplicadas
df["num_techniques"] = df[["pre", "in", "post"]].apply(
    lambda x: sum([1 for v in x if v != "none"]), axis=1
)

# Define marcador de acordo com o nº de técnicas
def get_marker(num):
    if num == 0:
        return "x"  # baseline
    elif num == 1:
        return "o"  # uma técnica
    elif num == 2:
        return "D"  # duas técnicas
    else:
        return "*"  # três técnicas

df["marker"] = df["num_techniques"].apply(get_marker)

# === Plot ===
plt.figure(figsize=(10,6))

for _, row in df.iterrows():
    plt.scatter(
        row[label_performance], 
        row[label_discrimination], 
        marker=row["marker"], 
        s=120, alpha=0.8, edgecolor="black", label=row["marker"]
    )

plt.xlabel(label_discrimination)
plt.ylabel(label_performance)
plt.title(label_performance + " vs " + label_discrimination +" (" + model+")")
plt.grid(True, linestyle="--", alpha=0.5)

plt.axvline(x=1.0, color="red", linestyle="--", linewidth=1)

# Legenda manual (baseada no nº de técnicas)
import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([], [], color="black", marker="x", linestyle="None", markersize=10, label="Baseline (0 técnicas)"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=10, label="1 técnica"),
    mlines.Line2D([], [], color="black", marker="D", linestyle="None", markersize=10, label="2 técnicas"),
    mlines.Line2D([], [], color="black", marker="*", linestyle="None", markersize=12, label="3 técnicas"),
]
plt.legend(handles=legend_elements, title="Mitigação", loc="best")

plt.tight_layout()
plt.show()
