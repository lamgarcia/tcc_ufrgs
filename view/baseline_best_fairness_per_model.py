import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult_new_1.csv")
df = pd.read_csv(csv_path)

label_performance = "accuracy"
label_discrimination = "equal_opportunity_difference"# "disparate_impact"
no_discrimination_line = 0.0 #1.0

# Função para criar descrição detalhada do pipeline
def pipeline_description(row):
    return f"{row['model']} (pre:{row['pre']}, in:{row['in']}, post:{row['post']})"

# Define forma para cada modelo
model_list = df["model"].unique()
markers = ["o", "s", "^", "D", "v", "<", ">", "*"]  # expandir se houver mais modelos
model_marker_map = dict(zip(model_list, markers))

# Seleciona pontos de melhor fairness e baseline
best_points = []
for model_name, group in df.groupby("model"):
    # Melhor fairness (DI mais próximo de 1)
    idx_best_fairness = (group[label_discrimination] - 1).abs().idxmin()
    row_fairness = group.loc[idx_best_fairness].copy()
    row_fairness["best_type"] = "best discrimination"
    row_fairness["legend_label"] = pipeline_description(row_fairness)
    best_points.append(row_fairness)
    
    # Baseline (pre=in=post=none)
    baseline_row = group[
        (group["pre"] == "none") & 
        (group["in"] == "none") & 
        (group["post"] == "none")
    ]
    if not baseline_row.empty:
        row_baseline = baseline_row.iloc[0].copy()
        row_baseline["best_type"] = "baseline"
        row_baseline["legend_label"] = pipeline_description(row_baseline)
        best_points.append(row_baseline)

best_df = pd.DataFrame(best_points).drop_duplicates()

# === Plot ===
plt.figure(figsize=(12,7))

# Plota linhas conectando pontos do mesmo modelo
for model_name, group in best_df.groupby("model"):
    plt.plot(
        group[label_discrimination],
        group[label_performance],
        linestyle="--",
        color="gray",
        alpha=0.5
    )

# Plota os pontos
for _, row in best_df.iterrows():
    marker_shape = model_marker_map[row["model"]]
    if row["best_type"] == "baseline":
        facecolor = "blue"
    else:  # best discrimination
        facecolor = "white"
    
    plt.scatter(
        row[label_discrimination],
        row[label_performance],
        marker=marker_shape,
        s=100,
        facecolors=facecolor,
        edgecolors="black",
        alpha=0.9,
        label=row["legend_label"]
    )

plt.xlabel(label_discrimination)
plt.ylabel(label_performance)
plt.title(f"{label_performance} vs {label_discrimination} [baseline (azul) e melhor fairness do modelo (branco)]")
plt.grid(True, linestyle="--", alpha=0.5)

# Linha de perfeito balanço (DI = 1)
plt.axvline(x=no_discrimination_line, color="red", linestyle="--", linewidth=1)

# Remove duplicados na legenda
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

# Legenda fora do gráfico à direita
plt.legend(
    by_label.values(), by_label.keys(),
    title="Modelo e mitigação",
    loc="lower left",
    #bbox_to_anchor=(1, 0.5),
    fontsize=9
)

plt.tight_layout()
plt.show()
