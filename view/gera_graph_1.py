import pandas as pd
import matplotlib.pyplot as plt

# === Lê o CSV ===
df = pd.read_csv("runs.csv")

# Cria identificador único
df["pipeline"] = df["model"].astype(str) + "_" + df["pre"].astype(str) + "_" + df["in"].astype(str) + "_" + df["post"].astype(str)

# Métricas de fairness
metrics = {
    "Statistical Parity Difference": "statistical_parity_difference",
    "Disparate Impact": "disparate_impact",
    "Equality of Opportunity": "equal_opportunity_difference",
    "Equalized Odds": "equalized_odds_difference",
    "Predictive Value": "average_predictive_value_difference"
}

# Linha de igualdade perfeita (0 ou 1 dependendo da métrica)
perfect_values = {
    "Statistical Parity Difference": 0,
    "Disparate Impact": 1,
    "Equality of Opportunity": 0,
    "Equalized Odds": 0,
    "Sufficiency": 1
}

# === Cria grid de gráficos ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (title, col) in enumerate(metrics.items()):
    ax = axes[i]
    ax.scatter(df[col], df["accuracy"], alpha=0.7)
    
    # Adiciona linha de igualdade perfeita
    ax.axvline(x=perfect_values[title], color='red', linestyle='--', label='Perfeita igualdade')
    
    ax.set_xlabel(title)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} vs Accuracy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

# Remove o subplot vazio (5 gráficos em grade 2x3)
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()
