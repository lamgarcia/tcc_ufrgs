
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("runs_adult_10x_1_mean_std.csv")

nonestring = 'X'
method_map = {
    "none": nonestring,
    "reweighing": "REW",
    "disparate_impact_remover": "DIR",
    "equalized_odds_postprocessing": "EOP",
    "reject_option_classification": "REJOC"
}

method_map_extenso = {
    "none": nonestring,
    "reweighing": "Reweighing",
    "disparate_impact_remover": "Disparate Impact Remover",
    "equalized_odds_postprocessing": "Equalized Odds PostProcessing",
    "reject_option_classification": "Reject Option Classification"
}

model_map = {
    "none": "None",
    "logistic_regression": "Logistic Reg.",
    "neural_network": "Neural Net.",
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "bernoulli_nb": "Naive Bayes"
}


metric_map_old = {
    "none": "None",
    "accuracy": "Acurácia",
    "balanced_accuracy": "Acurácia Balanceada",
    "precision": "Precisão",
    "recall": "Recall",
    "f1-score": "F1-Score",
    "roc_auc": "ROC-AUC",
    "demographic_parity_ratio": "Disparate Impact",
    "equal_odds_ratio": "Equalized Odds ",
    "equal_opportunity_ratio": "Equal Opportunity",
    "statistical_parity_difference": "Statistical Parity Difference",
    "average_predictive_value_difference": "Avg. Predictive Value Difference",
    "average_odds_difference": "Avg. Odds Difference"

}

metric_map = {
    # Confusion matrix (globais)
    "true_positives": "VP",
    "true_negatives": "VN",
    "false_positives": "FP",
    "false_negatives": "FN",

    # Métricas clássicas de desempenho
    "accuracy": "Acurácia",
    "recall": "Recall",
    "specificity": "Especificidade (TNR)",
    "precision": "Precisão",
    "npv": "Valor Pred. Neg.(NPV)",
    "f1-score": "F1-Score",
    "balanced_accuracy": "Acurácia Balanceada",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "mcc": "Coef. de Matthews (MCC)",
    "log_loss": "Log Loss",
    "cohen_kappa_score": "Kappa de Cohen",
    "fbeta_score": "Fβ-Score",
    "jaccard_score": "Índice de Jaccard",
    "hamming_loss": "Hamming Loss",

    # Fairness – ratios
    "demographic_parity_ratio": "Disparate Impact",
    "equal_odds_ratio": "Equalized Odds Ratio",
    "predictive_parity_ratio": "Predictive Parity Ratio",
    "equal_opportunity_ratio": "Equal Opportunity Ratio",
    "tnr_ratio": "TNR Ratio",
    "fpr_ratio": "FPR Ratio",
    "npv_ratio": "NPV Ratio",
    "fnr_ratio": "FNR Ratio",
    "error_rate_ratio": "Error Rate Ratio",
    "false_positive_rate_ratio": "FPR Ratio",
    "false_discovery_rate_ratio": "FDR Ratio",
    "false_negative_rate_ratio": "FNR Ratio",
    "false_omission_rate_ratio": "FOR Ratio",

    # Fairness – diferenças
    "average_odds_difference": "Average Odds Difference (Separação)",
    "average_predictive_value_difference": "Avg. Predictive Value Difference (Suficiência)",
    "statistical_parity_difference": "Statistical Parity Difference (Independência)",
    "equalized_odds_difference": "Equalized Odds Diff.",
    "equal_opportunity_difference": "Equal Opportunity Diff.",
    "error_rate_difference": "Error Rate Diff.",
    "false_positive_rate_difference": "FPR Diff.",
    "false_discovery_rate_difference": "FDR Diff.",
    "false_negative_rate_difference": "FNR Diff.",
    "false_omission_rate_difference": "FOR Diff.",
    "true_positive_rate_difference": "TPR Diff.",

    # Métricas de desigualdade entre grupos
    "between_group_coefficient_of_variation": "Coef. de Variação ",
    "generalized_entropy_index": "Índ. Entropia Generalizada",
    "differential_fairness_bias_amplification": "Differential Fairness",

    # Confusion matrix por grupo
    "tp_privileged": "TP (Priv)",
    "tp_unprivileged": "TP (Unpriv)",
    "tn_privileged": "TN (Priv)",
    "tn_unprivileged": "TN Unpriv)",
    "fn_privileged": "FN (Priv)",
    "fn_unprivileged": "FN (Unpriv)",
    "fp_privileged": "FP (Priv)",
    "fp_unprivileged": "FP (Unpriv)",

    # Métricas por grupo
    "accuracy_privileged": "Acurácia (Priv)",
    "accuracy_unprivileged": "Acurácia (Unpriv)",
    "recall_privileged": "Recall (Priv)",
    "recall_unprivileged": "Recall (Unpriv)",
    "specificity_privileged": "Especificidade (Priv)",
    "specificity_unprivileged": "Especificidade (Unpriv)",
    "precision_privileged": "Precisão (Male)",
    "precision_unprivileged": "Precisão (Female)",
    "npv_privileged": "NPV (Priv)",
    "npv_unprivileged": "NPV (Unpriv)",

    # Taxas por grupo
    "selection_rate_privileged": "Selection Rate (Priv)",
    "selection_rate_unprivileged": "Selection Rate (Unpriv)",
    "false_positive_rate_privileged": "FPR (Priv)",
    "false_positive_rate_unprivileged": "FPR (Unpriv)",
    "false_negative_rate_privileged": "FNR (Priv)",
    "false_negative_rate_unprivileged": "FNR (Unpriv)",
    "false_discovery_rate_privileged": "FDR (Priv)",
    "false_discovery_rate_uprivileged": "FDR (Unpriv)",
    "false_omission_rate_privileged": "FOR (Priv)",
    "false_omission_rate_uprivileged": "FOR (Unpriv)",
    "error_rate_privileged": "Error Rate (Priv)",
    "error_rate_unprivileged": "Error Rate (Unpriv)"
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
        return "Baseline"
def plot_front_paretto_global (metric_x, metric_y):
	# ============================================================
	# 1. Função de fronteira de Pareto
	# ============================================================
	def compute_pareto_front(df, metric_x, metric_y, maximize_y=True, target_x=1.0):
		"""
		Retorna os pontos da fronteira de Pareto.
		metric_x -> distância do fairness até o valor neutro (ex.: |DI - 1|)
		metric_y -> métrica de performance
		"""
		# Quanto menor a distância da fairness, melhor
		df = df.copy()
		df["fairness_dist"] = (df[metric_x] - target_x).abs()

		# Ordena por fairness → depois performance
		df_sorted = df.sort_values(by=["fairness_dist", metric_y], ascending=[True, not maximize_y])

		pareto = []
		best_performance_so_far = -np.inf

		for _, row in df_sorted.iterrows():
			perf = row[metric_y]
			if perf > best_performance_so_far:
				pareto.append(row)
				best_performance_so_far = perf

		return pd.DataFrame(pareto)


	def full_label(row):
		model_name = model_map.get(row["model"], row["model"])
		pre = method_map.get(row["pre"], row["pre"])
		post = method_map.get(row["post"], row["post"])
		mitig_label = f"{pre} + {post}"
		if pre=="X" and post=="X":
			mitig_label = "Baseline"
		return f"{model_name}\n{mitig_label}"


	# ============================================================
	# 3. Seleciona fronteira global
	# ============================================================

	pareto_df = compute_pareto_front(df, metric_x, metric_y, maximize_y=True, target_x=1.0)


	# ============================================================
	# 4. Plot da fronteira com balãozinho
	# ============================================================
	plt.figure(figsize=(12, 7))


	# Fronteira
	plt.plot(
		pareto_df[metric_x],
		pareto_df[metric_y],
		marker="o",
		linestyle="--",
		color="red",
		linewidth=2,
		label="Fronteira de Pareto (Global)"
	)

	# Pontos globais
	plt.scatter(df[metric_x], df[metric_y], color="gray", alpha=0.3, label="Todos os experimentos")

	# ------------------------------------------------------------
	#  Balõezinhos (anotações) em cada ponto da fronteira
	# ------------------------------------------------------------
	for _, row in pareto_df.iterrows():
		label = full_label(row)

		# Posição levemente deslocada para não colidir com o ponto
		ann = plt.annotate(
			label,
			(row[metric_x], row[metric_y]),
			textcoords="offset points",
			xytext=(10, -10),
			ha='left',
			fontsize=8,
			bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3")
		)

		ann.draggable()

  
	# Configurações
	plt.xlabel(metric_map[metric_x])
	plt.ylabel(metric_map[metric_y])
	#plt.title("Fronteira de Pareto (Global)\n")
	plt.xlim(0, 1.05)
	
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.show()



plot_front_paretto_global("demographic_parity_ratio","f1-score")
plot_front_paretto_global("equal_odds_ratio","f1-score")
