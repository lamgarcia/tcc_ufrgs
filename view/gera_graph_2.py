import pandas as pd
import matplotlib.pyplot as plt
import math

# === Lê o CSV ===
df = pd.read_csv("runs.csv")

# Cria identificador único
df["pipeline"] = df["model"].astype(str) + "_" + df["pre"].astype(str) + "_" + df["in"].astype(str) + "_" + df["post"].astype(str)

# Lista completa de métricas de fairness solicitadas
metrics = [
    "statistical_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "average_odds_difference",
    "error_rate_difference",
    "average_abs_odds_difference",
    "average_predictive_value_difference",
    "generalized_equalized_odds_difference",
    "disparate_impact",
    "error_rate_ratio",
    "generalized_entropy_index",
    "differential_fairness_bias_amplification"
]

# Valores de igualdade perfeita para algumas métricas
perfect_values = {
    "statistical_parity_difference": 0,
    "equalized_odds_difference": 0,
    "equal_opportunity_difference": 0,
    "average_odds_difference": 0,
    "error_rate_difference": 0,
    "average_abs_odds_difference": 0,
    "average_predictive_value_difference": 1,
    "generalized_equalized_odds_difference": 0,
    "disparate_impact": 1,
    "error_rate_ratio": 1,
}

# === Define grid dinâmica ===
num_metrics = len(metrics)
cols = 4  # número de colunas no grid
rows = math.ceil(num_metrics / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

# === Plota cada gráfico ===
for i, col in enumerate(metrics):
    ax = axes[i]
    ax.scatter(df[col], df["accuracy"], alpha=0.7)
    
    # linha de igualdade perfeita se existir
    if col in perfect_values:
        ax.axvline(x=perfect_values[col], color='red', linestyle='--', label='Perfeita igualdade')
        ax.legend()
    
    ax.set_xlabel(col)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{col} vs Accuracy")
    ax.grid(True, linestyle="--", alpha=0.5)

# Remove eixos extras, caso o grid seja maior que o número de métricas
for j in range(num_metrics, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
