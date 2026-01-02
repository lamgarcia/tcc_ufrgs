import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("runs_adult_30x_2_mean_std.csv")

nonestring = 'X'
method_map = {
    "none": nonestring,
    "reweighing": "Rew",
    "grid_search": "GridS",
    "disparate_impact_remover": "DImpRemov",
    "exponentiated_gradient": "ExpGrad",
    "equalized_odds_postprocessing": "EqOdds",
    "reject_option_classification": "RejOptClas"
}


model_map = {
    "none": "None",
    "bernoulli_nb": "Naive Bayes",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Reg",
    "neural_network": "Neural Net",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "xgboost": "XGBoost"
}

metric_map = {
    "none": "None",
    "accuracy": "Acurácia",
    "balanced_accuracy": "Acurácia Balanceada",
    "precision": "Precisão",
    "recall": "Recall",
    "f1-score": "F1-Score",
    "roc_auc": "ROC-AUC",
    "demographic_parity_ratio": "Disparate Impact",
    "equal_odds_ratio": "Equalized Odds Ratio",
    "equal_opportunity_ratio": "Equal Opportunity",
    "predictive_parity_ratio": "Predictive Parity"
}

markers_models = ["o", "s", "^", "D", "v", "<", ">", "*", "."]
markers_mitigation = ['o', 's', 'D', '^', 'v', 'P', 'X', '<', '>']
markers_metric = ['o', 's', '^', 'D', 'X', 'v', 'P', '*']

# === 1. Identifica o tipo de mitigação ===
def get_mitig_name(row):
    if row["pre"] != "none" and row["post"] == "none":
        return method_map.get(row["pre"], row["pre"])
    elif row["post"] != "none" and row["pre"] == "none":
        return method_map.get(row["post"], row["post"])
    elif row["pre"] != "none" and row["post"] != "none":
        pre_name = method_map.get(row["pre"], row["pre"])
        post_name = method_map.get(row["post"], row["post"])
        return f"{pre_name}+{post_name}"
    else:
        return "baseline"


def annotate_point(ax, x, y, text):
    """Cria um balão simples com seta, deslocado para cima e ARRATÁVEL."""
    ann = ax.annotate(
        text,
        (x, y),
        xytext=(10, 12),  # posição inicial
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=9
    )
    ann.draggable()  # <-- habilita arrastar com o mouse



def plot_model_scatter(ax, df, model_name, metric_x, metric_y):
    """Plota um modelo e conecta pontos à direita (X crescente) a partir do maior Y."""
    df["mitigation"] = df.apply(get_mitig_name, axis=1)
    df_model = df[df["model"] == model_name].copy()
    if df_model.empty:
        ax.text(0.5, 0.5, f"Sem dados para {model_name}", ha='center', va='center')
        return

    mitigs = df_model["mitigation"].unique()
    marker_map = {m: markers_mitigation[i % len(markers_mitigation)] for i, m in enumerate(mitigs)}

    # === 1. Plota os pontos ===
    for mitigation, group in df_model.groupby("mitigation", observed=True):
        ax.scatter(
            group[metric_x],
            group[metric_y],
            marker=marker_map[mitigation],
            s=100,
            label=mitigation,
            alpha=0.8
        )

    # === 2. Seleciona o ponto de maior Y e conecta só com X maiores ===
    sorted_by_y = df_model.sort_values(by=metric_y, ascending=False)
    x_vals = sorted_by_y[metric_x].values
    y_vals = sorted_by_y[metric_y].values
    mitig_vals = sorted_by_y["mitigation"].values

    x_start, y_start = x_vals[0], y_vals[0]
    selected_points = [(x_start, y_start, mitig_vals[0])]

    for i in range(1, len(x_vals)):
        if x_vals[i] > selected_points[-1][0]:
            selected_points.append((x_vals[i], y_vals[i], mitig_vals[i]))

    # === 3. Traçar linha da fronteira ===
    if len(selected_points) > 1:
        xs = [p[0] for p in selected_points]
        ys = [p[1] for p in selected_points]
        ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # === 4. Balões SOMENTE nos pontos da fronteira ===
    for x, y, mitig in selected_points:
        annotate_point(ax, x, y, mitig)

    # === 5. Personalização ===
    #ax.set_xlabel(metric_map.get(metric_x, metric_x))
    ax.set_xlabel(metric_map[metric_x])
    ax.set_ylabel(metric_map[metric_y])
    ax.set_title(f"Modelo: {model_map[model_name]}")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_model_scatter_grid(df, plot_list, max_cols=3):
    n_plots = len(plot_list)
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = np.array(axes).reshape(-1)

    for i, (model_name, metric_x, metric_y) in enumerate(plot_list):
        plot_model_scatter(axes[i], df, model_name, metric_x, metric_y)

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    

    handles, labels = axes[0].get_legend_handles_labels()
    '''fig.legend(
        handles, 
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),   # Ajuste vertical aqui
        ncol=9,
        fontsize=11,
        frameon=True,
        title="Mitigação"
    )'''
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.15)  # <-- reduz o espaço horizontal entre os subplots

    #name_file_compara = "img_pareto_criterios_" + model_name
    #plt.savefig(name_file_compara, dpi=300, bbox_inches="tight")

    plt.show()



# Execução
plot_model_scatter_grid(df, [
    ("svm", "equal_odds_ratio", "demographic_parity_ratio"),
    ("svm", "predictive_parity_ratio", "demographic_parity_ratio"),
    ("svm", "predictive_parity_ratio", "equal_odds_ratio")
])
