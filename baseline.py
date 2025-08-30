import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss, confusion_matrix
)
from sklearn.datasets import fetch_openml
from datetime import datetime
import uuid

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# ========= 1. Carregar e preparar dados =========
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame.rename(columns={'class': 'income'})
df.dropna(inplace=True)
df = df[df['sex'].isin(['Male', 'Female'])]

# Codificar target e atributo sensível
df['income_bin'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
df['sex_bin'] = LabelEncoder().fit_transform(df['sex'])  # 0=Female, 1=Male

X = df.drop(['income', 'income_bin', 'sex', 'sex_bin'], axis=1)
y = df['income_bin']
A = df['sex_bin']

# One-hot e standard scaler
X_encoded = pd.get_dummies(X, drop_first=True)
X_scaled = StandardScaler().fit_transform(X_encoded)

# ========= 2. Divisão dos dados =========
X_temp, X_test, y_temp, y_test, A_temp, A_test = train_test_split(
    X_scaled, y, A, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, A_train, A_val = train_test_split(
    X_temp, y_temp, A_temp, test_size=0.25, random_state=42)

# ========= 3. Treinar modelo =========
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
model_name = "LogisticRegression"

# ========= 4. Avaliação =========
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]


model_info = pd.DataFrame([{
    "id": str(uuid.uuid4()),
    "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": model_name,
}])


# ===== PERFORMANCE 

tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
specificity = tn / (tn + fp)
balanced_acc = balanced_accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_proba)
pr_auc = average_precision_score(y_val, y_proba)
mcc = matthews_corrcoef(y_val, y_pred)
logloss = log_loss(y_val, y_proba)
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

performance_metrics = pd.DataFrame([{
    "acuracia": accuracy,
    "precisao": precision,
    "npv": npv,
    "recall": recall,
    "specificity": specificity,
    "f1-score": f1,
    "balanced_accuracy": balanced_acc,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "mcc": mcc,
    "log_loss": logloss,
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn
}])

#### FAIRNESS

tn0, fp0, fn0, tp0 = confusion_matrix(y_val[A_val==0], y_pred[A_val==0]).ravel()
tn1, fp1, fn1, tp1 = confusion_matrix(y_val[A_val==1], y_pred[A_val==1]).ravel()

dataset_true = BinaryLabelDataset(
    df=pd.DataFrame({"income": y_val, "sex": A_val}),
    label_names=['income'],
    protected_attribute_names=['sex'],
    favorable_label=1,
    unfavorable_label=0
)

dataset_pred = dataset_true.copy()
dataset_pred.labels = y_pred.reshape(-1, 1)

metric = ClassificationMetric(
    dataset_true,
    dataset_pred,
    privileged_groups=[{'sex': 1}],
    unprivileged_groups=[{'sex': 0}]
)

dpd = metric.statistical_parity_difference()
eod = metric.equalized_odds_difference()
aod = metric.average_odds_difference()
# ========= 5b. Predictive Parity =========
ppv0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0  # grupo não privilegiado
ppv1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0.0  # grupo privilegiado
predictive_parity_diff = ppv1 - ppv0

fairness_metrics = pd.DataFrame([{
    "statistical_parity_diff": dpd,
    "equalized_odds_diff": eod,
    "average_odds_diff": aod,
    "predictive_parity_diff": predictive_parity_diff
}])




runs_file = "runs.csv"
results = pd.concat([model_info, performance_metrics, fairness_metrics], axis=1)

if not os.path.exists(runs_file):
    results.to_csv(runs_file, index=False)
else:
    results.to_csv(runs_file, index=False, mode="a", header=False)

print("Métricas salvas em 'runs.csv'")
print(results)
