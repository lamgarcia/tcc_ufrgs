import os
import uuid
from datetime import datetime

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

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def load_and_preprocess():
    """Carrega e prepara o dataset Adult do OpenML."""
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame.rename(columns={'class': 'income'})

    # limpeza
    df.dropna(inplace=True)
    df = df[df['sex'].isin(['Male', 'Female'])]

    # binarização
    df['income_bin'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    df['sex_bin'] = LabelEncoder().fit_transform(df['sex'])  # 0=Female, 1=Male

    # features, target e atributo sensível
    X = df.drop(['income', 'income_bin', 'sex', 'sex_bin'], axis=1)
    y = df['income_bin']
    A = df['sex_bin']

    # one-hot e escala
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_scaled = StandardScaler().fit_transform(X_encoded)

    return X_scaled, y, A


def split_data(X, y, A, test_size=0.2, val_size=0.25, random_state=42):
    """Divide os dados em treino, validação e teste."""
    X_temp, X_test, y_temp, y_test, A_temp, A_test = train_test_split(
        X, y, A, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val, A_train, A_val = train_test_split(
        X_temp, y_temp, A_temp, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test


def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    model_name = "LogisticRegression"
    return model, model_name


def evaluate_performance(y_true, y_pred, y_proba):
    """Calcula métricas de performance."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return pd.DataFrame([{
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "recall": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1-score": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }])


def evaluate_fairness(y_true, y_pred, A, protected_attribute, label):
   
    dataset_true = BinaryLabelDataset(
        df=pd.DataFrame({label: y_true.values, protected_attribute: A.values}),
        label_names=[label],
        protected_attribute_names=[protected_attribute],
        favorable_label=1,
        unfavorable_label=0
    )
    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)

    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        privileged_groups=[{protected_attribute: 1}],
        unprivileged_groups=[{protected_attribute: 0}]
    )
 
    return pd.DataFrame([{
        "statistical_parity_diff": metric.statistical_parity_difference(),
        "equalized_odds_diff": metric.equalized_odds_difference(),
        "average_odds_diff": metric.average_odds_difference(),
        "disparate_impact": metric.disparate_impact(),
  
        "positive_predictive_value": metric.positive_predictive_value()
    }])


def main():
    X, y, A = load_and_preprocess()
    X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = split_data(X, y, A)

    model, model_name = train_model(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    model_info = pd.DataFrame([{
        "id": str(uuid.uuid4()),
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
    }])

    mitigation = pd.DataFrame([{"pre": "none", "in": "none", "post": "none"}])
    
    performance_metrics = evaluate_performance(y_val, y_pred, y_proba)
    
    fairness_metrics = evaluate_fairness(y_val, y_pred, A_val, protected_attribute = 'sex', label = 'income')

    results = pd.concat(
        [model_info, mitigation, performance_metrics, fairness_metrics],
        axis=1
    )

    results_file="runs.csv"
    
    if os.path.exists(results_file):
        results.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results.to_csv(results_file, index=False)

    print("Save:", results_file)
    print(results)


if __name__ == "__main__":
    main()
