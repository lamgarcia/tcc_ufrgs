import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult.csv")
df = pd.read_csv(csv_path)

# Métricas de interesse
metrics = [
    #"accuracy",
    #"precision",
    #"recall",
    #"specificity",
    #"f1-score",
    #"roc_auc"
    "disparate_impact",
    "predictive_parity_ratio",
    "equal_opportunity_ratio"
]

# Dicionário de abreviações

nonestring = 'X'
method_map = {
    "none": nonestring,
    "reweighing": "REW",
    "grid_search": "GRIDS",
    "disparate_impact_remover": "DISPIMP",
    "exponentiated_gradient": "EXGRAD",
    "equalized_odds_postprocessing": "EQODDS",
    "reject_option_classification": "REJECTOPT",
}

# Aplica substituição dos nomes longos para curtos
df = df.replace(method_map)

# Identifica baseline
baseline = df[(df['pre'] == nonestring) & (df['in'] == nonestring) & (df['post'] ==nonestring)]

# Função para atribuir prioridade de ordenação
def pipeline_order(row):
    pre, inn, post = row["pre"], row["in"], row["post"]
    if pre == nonestring and inn == nonestring and post == nonestring:
        return 0  # baseline
    if pre != nonestring and inn == nonestring and post == nonestring:
        return 1  # só pre
    if pre == nonestring and inn != nonestring and post == nonestring:
        return 2  # só in
    if pre == nonestring and inn == nonestring and post != nonestring:
        return 3  # só post
    if pre != nonestring and inn != nonestring and post == nonestring:
        return 4  # pre+in
    if pre == nonestring and inn != nonestring and post != nonestring:
        return 5  # in+post
    if pre != nonestring and inn != nonestring and post != nonestring:
        return 6  # pre+in+post
    return 99  # fallback

# Lista de modelos
models = df['model'].unique()

for model in models:
    # Subset para o modelo
    df_model = df[df['model'] == model].copy()
    baseline_model = baseline[baseline['model'] == model]
    
    if baseline_model.empty:
        print(f"Atenção: baseline não encontrado para o modelo {model}")
        continue
    
    # Pegando apenas as métricas do baseline
    baseline_values = baseline_model[metrics].iloc[0]
    
    # Calcula delta percentual para cada linha do modelo
    df_ratio = df_model.copy()
    for m in metrics:
        df_ratio[m] = (df_ratio[m] - baseline_values[m]) / baseline_values[m]
    
    # Cria coluna pipeline e ordem
    df_ratio['pipeline'] = df_ratio['pre'] + "-" + df_ratio['in'] + "-" + df_ratio['post']
    df_ratio['order'] = df_ratio.apply(pipeline_order, axis=1)
    
    # Reorganiza dados para heatmap respeitando a ordem
    heatmap_data = df_ratio.sort_values("order").set_index('pipeline')[metrics].T
    
    # === Plota heatmap ===
    plt.figure(figsize=(10, len(metrics)*0.6 + 2))
    ax = sns.heatmap(
        heatmap_data, annot=True, cmap="RdBu", center=0, fmt=".2%", cbar=True
    )
    
    # Formata a barra de cores em porcentagem
    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Ajusta os rótulos do eixo X
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    
    plt.title(f"Δ relativo ao baseline (%) - Modelo: {model}")
    plt.xlabel("Pipeline (pre-in-post)")
    plt.ylabel("Métrica")
    plt.tight_layout()
    plt.show()

