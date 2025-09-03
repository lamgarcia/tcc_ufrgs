import os
import uuid
import yaml
from datetime import datetime
import importlib
import numpy as np

import pandas as pd
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

# ===================== Config Loader =====================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ===================== Data Preprocessing =====================

def load_and_preprocess(dataset_cfg):
    #adult = fetch_openml(name=dataset_cfg["name"], version=dataset_cfg["version"], as_frame=True)
    #df = adult.frame.rename(columns={"class": dataset_cfg["target"]})
    #df.dropna(inplace=True)

    # lê o CSV local
    df = pd.read_csv(dataset_cfg["path"])

    # garante que o nome da coluna alvo está certo
    if dataset_cfg["target"] not in df.columns:
        raise ValueError(f"Coluna alvo '{dataset_cfg['target']}' não encontrada no dataset.")

    # remove linhas com valores ausentes
    df.dropna(inplace=True)  # ou use imputação
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Filtra apenas categorias privilegiadas e não privilegiadas
    sensitive_col = dataset_cfg["sensitive"]
    privileged = dataset_cfg["privileged"]
    unprivileged = dataset_cfg["unprivileged"]
    df = df[df[sensitive_col].isin(privileged + unprivileged)]
    target_col = dataset_cfg["target"]
    favorable = dataset_cfg["favorable"]  
    unfavorable  = dataset_cfg["unfavorable"] 
        
    # Cria coluna binária para atributo sensível
    df[f"{sensitive_col}_bin"] = df[sensitive_col].apply(
        lambda x: 1 if x in privileged else 0
    )

    # Cria coluna binária para o target
    df[f"{target_col}_bin"] = df[target_col].apply(
        lambda x: 1 if x == favorable else 0
    )

    # Features e target
    #y = df[dataset_cfg["target"]]
    y = df[f"{target_col}_bin"]
    A = df[f"{sensitive_col}_bin"]
    X = df.drop([dataset_cfg["target"], f"{target_col}_bin", sensitive_col, f"{sensitive_col}_bin"], axis=1)

    # One-hot encoding e padronização
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_scaled = StandardScaler().fit_transform(X_encoded)

    return X_scaled, y, A


def split_data(X, y, A, split_cfg):
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, A,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"]
    )

    return X_train, X_test, y_train, y_test, A_train, A_test

# ===================== Model Training =====================
def train_model(model_cfg, X_train, y_train, sample_weight=None):
    module_path = f"src.models.{model_cfg['name'].lower()}"
    model_module = importlib.import_module(module_path)
    model = model_module.create_model(model_cfg.get("params", {}))
    
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    
    return model, model_cfg["name"]

# ===================== Mitigation =====================
def apply_mitigation_pre(X_train, y_train, A_train, mitigation_cfg):
    if mitigation_cfg["name"].lower() == "none":
        return X_train, y_train, None  # sempre 3 valores

    module_path = f"src.mitigation.pre.{mitigation_cfg['name'].lower()}"
    mitigation_module = importlib.import_module(module_path)
    return mitigation_module.apply(X_train, y_train, A_train, mitigation_cfg.get("params", {}))

def apply_mitigation_in(model, X_train, y_train, A_train, mitigation_cfg):
    if mitigation_cfg["name"].lower() == "none":
        return model
    module_path = f"src.mitigation.in_processing.{mitigation_cfg['name'].lower()}"
    mitigation_module = importlib.import_module(module_path)
    return mitigation_module.apply(model, X_train, y_train, A_train, mitigation_cfg.get("params", {}))

def apply_mitigation_post(y_pred, y_proba, y_test, A_test, mitigation_cfg):
    if mitigation_cfg["name"].lower() == "none":
        return y_pred, y_proba
    module_path = f"src.mitigation.post.{mitigation_cfg['name'].lower()}"
    mitigation_module = importlib.import_module(module_path)
    return mitigation_module.apply(y_pred, y_proba, y_test, A_test, mitigation_cfg.get("params", {}))

# ===================== Metrics =====================
def evaluate_performance(y_true, y_pred, y_proba):
    # Descobre automaticamente os rótulos únicos
    unique_labels = pd.Series(y_true).dropna().unique()
    if len(unique_labels) != 2:
        raise ValueError(f"Target não binário: {unique_labels}")

    # Cria mapeamento para 0 e 1
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    y_true_num = pd.Series(y_true).map(label_map).astype(int)
    y_pred_num = pd.Series(y_pred).map(label_map).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_num, y_pred_num, labels=[0,1]).ravel()
    return pd.DataFrame([{
        "acuracia": accuracy_score(y_true_num, y_pred_num),
        "precisao": precision_score(y_true_num, y_pred_num),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "recall": recall_score(y_true_num, y_pred_num),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1-score": f1_score(y_true_num, y_pred_num),
        "balanced_accuracy": balanced_accuracy_score(y_true_num, y_pred_num),
        "roc_auc": roc_auc_score(y_true_num, y_proba),
        "pr_auc": average_precision_score(y_true_num, y_proba),
        "mcc": matthews_corrcoef(y_true_num, y_pred_num),
        "log_loss": log_loss(y_true_num, y_proba),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }])


def evaluate_fairness(y_true, y_pred, A, protected_attribute, label):
    # Converte y_true e y_pred para 0/1
    unique_labels = pd.Series(y_true).dropna().unique()
    if len(unique_labels) != 2:
        raise ValueError(f"Target não binário: {unique_labels}")
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    y_true_num = pd.Series(y_true).map(label_map).astype(int)
    y_pred_num = pd.Series(y_pred).map(label_map).astype(int)

    # Converte atributo protegido para 0/1 (caso ainda não seja numérico)
    A_num = pd.Series(A)
    if not pd.api.types.is_numeric_dtype(A_num):
        unique_attrs = A_num.dropna().unique()
        if len(unique_attrs) != 2:
            raise ValueError(f"Atributo protegido não binário: {unique_attrs}")
        attr_map = {unique_attrs[0]: 0, unique_attrs[1]: 1}
        A_num = A_num.map(attr_map).astype(int)

    dataset_true = BinaryLabelDataset(
        df=pd.DataFrame({label: y_true_num, protected_attribute: A_num}),
        label_names=[label],
        protected_attribute_names=[protected_attribute],
        favorable_label=1,
        unfavorable_label=0
    )

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred_num.values.reshape(-1, 1)

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

# ===================== Run Experiment =====================
def run_experiment(config_path):
    config = load_config(config_path)

    # === Data ===
    X, y, A = load_and_preprocess(config["dataset"])
    X_train, X_test, y_train, y_test, A_train, A_test = split_data(X, y, A, config["split"])

    # === Pre-processing mitigation ===
    X_train, y_train, sample_weight = apply_mitigation_pre(X_train, y_train, A_train, config["mitigation"]["pre"])

    # === Model ===
    model, model_name = train_model(config["model"], X_train, y_train, sample_weight=sample_weight)

    # === In-processing mitigation ===
    model = apply_mitigation_in(model, X_train, y_train, A_train, config["mitigation"]["in"])

    # === Test ===
    if config["mitigation"]["in"]["name"] == 'none': # hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print('nao usei in mitigation')
    else:
        print(' usei in mitigation')
        # Se o modelo for AIF360 in-processing
        df_test = pd.DataFrame(X_test)
        df_test['label'] = y_test.values
        df_test['protected'] = y_test.values
        dataset_test = BinaryLabelDataset(
            df=df_test,
            label_names=['label'],
            protected_attribute_names=['protected'],
            favorable_label=1,
            unfavorable_label=0
        )
        y_pred = model.predict(dataset_test).labels.ravel()
        y_proba = y_pred  # AIF360 não retorna probabilidade, apenas labels (ruim para log-loss, auc, roc..)
                          # Evitar AIF360 se você precisa de probabilidades calibradas — use modelos fairness-aware que integram com sklearn (como fairlearn).
                          #   
    # === Post-processing mitigation ===
    y_pred, y_proba = apply_mitigation_post(y_pred, y_proba, y_test, A_test, config["mitigation"]["post"])

    # === Info e Metrics ===
    model_info = pd.DataFrame([{
        "id": str(uuid.uuid4()),
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name
    }])

    mitigation_info = pd.DataFrame([{
        "pre": config["mitigation"]["pre"]["name"],
        "in": config["mitigation"]["in"]["name"],
        "post": config["mitigation"]["post"]["name"]
    }])

    performance_metrics = evaluate_performance(y_test, y_pred, y_proba)

    fairness_metrics = evaluate_fairness(
        y_test, y_pred, A_test,
        protected_attribute=config["dataset"]["sensitive"],
        label=config["dataset"]["target"]
    )

    results = pd.concat([model_info, mitigation_info, performance_metrics, fairness_metrics], axis=1)

    # === Save results ===
    results_file = "runs.csv"
    if os.path.exists(results_file):
        results.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results.to_csv(results_file, index=False)

    print("Save:", results_file)
    print(results)

    return results

run_experiment(config_path="config.yaml")