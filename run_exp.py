import os
import time
import sys
import uuid
import yaml
import importlib

import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef, log_loss, confusion_matrix,  cohen_kappa_score, fbeta_score, jaccard_score, hamming_loss)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from fairlearn.metrics import equalized_odds_ratio, equal_opportunity_ratio, demographic_parity_ratio
 
from sklearnex import patch_sklearn
patch_sklearn()

# ===================== Config Loader =====================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ===================== Data Preprocessing =====================

def load_and_preprocess(dataset_cfg, path_dataset):

    df = pd.read_csv(path_dataset)

    # garante que o nome da coluna alvo está certo
    if dataset_cfg["target"] not in df.columns:
        raise ValueError(f"Target column '{dataset_cfg['target']}' not found in dataset.")

    # remove linhas com valores ausentes
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.dropna(inplace=True)
    
    # Filtra apenas categorias privilegiadas e não privilegiadas
    sensitive_col = dataset_cfg["sensitive"]
    privileged    = dataset_cfg["privileged"]
    unprivileged  = dataset_cfg["unprivileged"]
    target_col    = dataset_cfg["target"]
    favorable     = dataset_cfg["favorable"]  
    unfavorable   = dataset_cfg["unfavorable"] 
    df = df[df[sensitive_col].isin(privileged + unprivileged)]
        
    # Cria coluna binária para atributo sensível
    #df[f"{sensitive_col}_bin"] = df[sensitive_col].apply(
    df[f"protected_bin"] = df[sensitive_col].apply(
         
        lambda x: 1 if x in privileged else 0
    )

    # Cria coluna binária para o target
    #df[f"{target_col}_bin"] = df[target_col].apply(
    df[f"label_bin"] = df[target_col].apply(         
        lambda x: 1 if x == favorable else 0
    )

    # Features e target
    #y = df[dataset_cfg["target"]]
    y = df[f"label_bin"]
    A = df[f"protected_bin"]
    X = df.drop([dataset_cfg["target"], f"label_bin", sensitive_col, f"protected_bin"], axis=1)

    # One-hot encoding 
    X = pd.get_dummies(X, drop_first=True)

    X.columns = (
        X.columns.astype(str)
        .str.replace("[", "(", regex=False)
        .str.replace("]", ")", regex=False)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    A = A.reset_index(drop=True)
 
    return X, y, A


#def split_data(X, y, A, split_cfg):
#    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
#        X, y, A,
#        test_size=split_cfg["test_size"],
#        random_state=split_cfg["random_state"]
#    )
#
#    return X_train, X_test, y_train, y_test, A_train, A_test

def _get_valid_fit_params(fit_method, params):
    """
    Filtra apenas os parâmetros válidos para o método fit() do modelo.
    """
    import inspect
    sig = inspect.signature(fit_method)
    valid_params = set(sig.parameters.keys())
    
    # Ignorar parâmetros especiais como *args, **kwargs
    valid_params.discard("args")
    valid_params.discard("kwargs")
    
    # Retorna apenas os parâmetros que o fit() aceita
    return {k: v for k, v in params.items() if k in valid_params}

# ===================== Model Training =====================

def train_model(model_name, X_train, y_train, params_model, params_fit=None):
    
    if params_model is None:
        params_model = {}

    if params_fit is None:
        params_fit = {}

    module_path = f"src.models.{model_name}"
    try:
        model_module = importlib.import_module(module_path)
        model = model_module.create_model(params_model)
        #model_cfg.get("params", {})
    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' not found in src.models")
        
    # Chama fit com todos os params possíveis (filtrando apenas os válidos)
    fit_params = _get_valid_fit_params(model.fit, params_fit)
    
    model.fit(X_train, y_train, **fit_params)
    return model

# ===================== Mitigation =====================
def apply_mitigation_pre(X_train, y_train, A_train, mitigation_cfg):
    
    if mitigation_cfg["name"].lower() == "none":
        return X_train, y_train, A_train, {}  
    try:
        module_path = f"src.mitigation.pre.{mitigation_cfg['name'].lower()}"
        mitigation_module = importlib.import_module(module_path)

        X_train, y_train, A_train, params = mitigation_module.apply(
            X_train, y_train, A_train, mitigation_cfg.get("params", {})
        )

        if params is None:  # params is dict
            params = {}

        return X_train, y_train, A_train, params

    except ModuleNotFoundError:
        raise ValueError(f"Mitigation method '{mitigation_cfg['name']}' not found in '{module_path}'")
    except AttributeError:
            raise ValueError(f"The module {module_path} must have a function 'apply(X_train, y_train, A_train, params)'")
    
def apply_mitigation_in(model, X_train, y_train, A_train, mitigation_cfg):
    
    if mitigation_cfg["name"].lower() == "none":
        return model   
    try:
        module_path = f"src.mitigation.in.{mitigation_cfg['name'].lower()}"
        mitigation_module = importlib.import_module(module_path)
        return mitigation_module.apply(model, X_train, y_train, A_train, mitigation_cfg.get("params", {}))

    except ModuleNotFoundError:
            raise ValueError(f"Mitigation method '{mitigation_cfg['name']}' not found in '{module_path}'")
    except AttributeError:
            raise ValueError(f"The module {module_path} must have a function 'apply(model, X_train, y_train, A_train, params)'")

def apply_mitigation_post(y_pred, y_proba, y_test, A_test, mitigation_cfg):

    if mitigation_cfg["name"].lower() == "none":
        return y_pred, y_proba
    try:
        module_path = f"src.mitigation.post.{mitigation_cfg['name'].lower()}"
        mitigation_module = importlib.import_module(module_path)
        return mitigation_module.apply(y_pred, y_proba, y_test, A_test, mitigation_cfg.get("params", {}))
    except ModuleNotFoundError:
            raise ValueError(f"Mitigation method '{mitigation_cfg['name']}' not found in '{module_path}'")
    except AttributeError:
            raise ValueError(f"The module {module_path} must have a function 'apply(y_pred, y_proba, y_test, A_test, params)'")
    
# ===================== Metrics =====================
def evaluate_performance(y_true, y_pred, y_proba):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp)  > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_score    = (2 * precision * recall) / (precision + recall) if (precision + recall)  > 0 else 0.0
    balancedaccuracy = (recall + specificity) / 2

    return pd.DataFrame([{
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy,
        "recall": recall,        
        "specificity": specificity,
        "precision": precision,
        "npv"     : npv,      
        "f1-score": f1_score,
        "balanced_accuracy": balancedaccuracy,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "cohen_kappa_score": cohen_kappa_score(y_true, y_pred),
        "fbeta_score": fbeta_score(y_true, y_pred, beta=2),
        "jaccard_score": jaccard_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }])


def evaluate_fairness(y_true, y_pred, A, sensitive_attribute, target):

    dataset_true = BinaryLabelDataset(
        df=pd.DataFrame({target: y_true, sensitive_attribute: A}),
        label_names=[target],
        protected_attribute_names=[sensitive_attribute],
        favorable_label=1,
        unfavorable_label=0
    )

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred

    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        privileged_groups=[{sensitive_attribute: 1}],
        unprivileged_groups=[{sensitive_attribute: 0}]
    )

    pg_value = True  # True for privileged_groups
    ug_value = False # False para unprivileged_groups

    fpr_priv = metric.false_positive_rate(privileged=pg_value)
    fpr_unpriv = metric.false_positive_rate(privileged=ug_value)
    fpr_ratio = (fpr_priv / fpr_unpriv) if fpr_unpriv > 0 else None

    fnr_priv = metric.false_negative_rate(privileged=pg_value)
    fnr_unpriv =  metric.false_negative_rate(privileged=ug_value)
    fnr_ratio = fnr_priv / fnr_unpriv if fnr_unpriv > 0 else None    

    tpr_priv = metric.recall(privileged=pg_value)
    tpr_unpriv = metric.recall(privileged=ug_value)
    equalized_opportunity_ratio = tpr_priv / tpr_unpriv if tpr_unpriv > 0 else None    #tpr_ratio

    tnr_priv =  metric.specificity(privileged=pg_value)
    tnr_unpriv =  metric.specificity(privileged=ug_value)
    tnr_ratio = (tnr_priv / tnr_unpriv) if tnr_unpriv > 0 else None

    equal_odds_ratio = (equalized_opportunity_ratio + fpr_ratio) / 2

    npv_priv = metric.negative_predictive_value(privileged=pg_value)
    npv_unpriv = metric.negative_predictive_value(privileged=ug_value)
    npv_ratio =  (npv_priv / npv_unpriv) if npv_unpriv > 0 else None  

    ppv_priv = metric.positive_predictive_value(privileged=pg_value) # precision
    ppv_unpriv = metric.positive_predictive_value(privileged=ug_value)
    #predictive_parity_ratio = ppv_priv / ppv_unpriv if ppv_unpriv > 0 else None
    if ppv_priv == 0 or ppv_unpriv == 0:
        predictive_parity_ratio = 0.0
    else:
        predictive_parity_ratio = min(ppv_priv / ppv_unpriv, ppv_unpriv / ppv_priv)

    fairlearn_args = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive_features": A,
        "method": "between_groups"
    }

   
    return pd.DataFrame([{
        #ratios

        "disparate_impact": demographic_parity_ratio(**fairlearn_args),       #metric.disparate_impact()
        "equal_odds_ratio": equalized_odds_ratio(**fairlearn_args, agg='worst_case'),                                                   
        "predictive_parity_ratio":  predictive_parity_ratio,
        "equal_opportunity_ratio": equal_opportunity_ratio(**fairlearn_args),
        "tnr_ratio": tnr_ratio,
        "fpr_ratio": fpr_ratio,
        "npv_ratio": npv_ratio,
        "fnr_ratio": fnr_ratio,
        "error_rate_ratio": metric.error_rate_ratio(),
        "average_odds_difference": metric.average_odds_difference(),
        "average_predictive_value_difference": metric.average_predictive_value_difference(),
        "false_positive_rate_ratio": metric.false_positive_rate_ratio(),        
        "false_discovery_rate_ratio": metric.false_discovery_rate_ratio(),
        "false_negative_rate_ratio": metric.false_negative_rate_ratio(),
        "false_omission_rate_ratio": metric.false_omission_rate_ratio(),

        # difference
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "equalized_odds_difference": metric.equalized_odds_difference(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "error_rate_difference": metric.error_rate_difference(),
        "false_positive_rate_difference": metric.false_positive_rate_difference(),
        "false_discovery_rate_difference": metric.false_discovery_rate_difference(),
        "false_negative_rate_difference": metric.false_negative_rate_difference(),
        "false_omission_rate_difference": metric.false_omission_rate_difference(),
        "true_positive_rate_difference": metric.true_positive_rate_difference(),

        # between groups
        "between_all_groups_coefficient_of_variation": metric.between_all_groups_coefficient_of_variation(),
        "between_all_groups_generalized_entropy_index": metric.between_all_groups_generalized_entropy_index(alpha=2),
        "between_group_generalized_entropy_index": metric.between_group_generalized_entropy_index(alpha=2),
        "between_all_groups_theil_index": metric.between_all_groups_theil_index(),
        "between_group_coefficient_of_variation": metric.between_group_coefficient_of_variation(),
        "between_group_theil_index": metric.between_group_theil_index(),  
        "generalized_entropy_index": metric.generalized_entropy_index(),
        "differential_fairness_bias_amplification": metric.differential_fairness_bias_amplification(concentration=1.0),
        
        # privileged x unprivileged
        "tp_privileged": metric.num_true_positives(privileged=pg_value),
        "tp_unprivileged": metric.num_true_positives(privileged=ug_value),
        "tn_privileged": metric.num_true_negatives(privileged=pg_value),
        "tn_unprivileged": metric.num_true_negatives(privileged=ug_value),
        "fn_privileged": metric.num_false_negatives(privileged=pg_value),
        "fn_unprivileged": metric.num_false_negatives(privileged=ug_value),
        "fp_privileged": metric.num_false_positives(privileged=pg_value),
        "fp_unprivileged": metric.num_false_positives(privileged=ug_value),                        
        "accuracy_privileged": metric.accuracy(privileged=pg_value),
        "accuracy_unprivileged": metric.accuracy(privileged=ug_value),
        "recall_privileged":        tpr_priv,
        "recall_unprivileged":      tpr_unpriv,
        "specificity_privileged":   tnr_priv,
        "specificity_unprivileged": tnr_unpriv,    
        "precision_privileged":     ppv_priv,
        "precision_unprivileged":   ppv_unpriv,
        "npv_privileged": npv_priv,
        "npv_unprivileged": npv_unpriv,

        "selection_rate_privileged": metric.selection_rate(privileged=pg_value),
        "selection_rate_unprivileged": metric.selection_rate(privileged=ug_value),
        "false_positive_rate_privileged": fpr_priv,
        "false_positive_rate_unprivileged": fpr_unpriv,
        "false_negative_rate_privileged": fnr_priv,
        "false_negative_rate_unprivileged": fnr_unpriv,
        "false_discovery_rate_privileged": metric.false_discovery_rate(privileged=pg_value),
        "false_discovery_rate_uprivileged": metric.false_discovery_rate(privileged=ug_value),
        "false_omission_rate_privileged": metric.false_omission_rate(privileged=pg_value),
        "false_omission_rate_uprivileged": metric.false_omission_rate(privileged=ug_value),        
        "error_rate_privileged": metric.error_rate(privileged=pg_value),
        "error_rate_unprivileged": metric.error_rate(privileged=ug_value)
    }])

# ===================== Run Experiment =====================
def run_experiment(config_path):

    start_time = time.time()

    config = load_config(config_path)

    # === Data ===
    # full
    #X, y, A = load_and_preprocess(config["dataset"], config["dataset"]["path"])
    #X_train, X_test, y_train, y_test, A_train, A_test = split_data(X, y, A, config["split"])
    
    # train and test
    X_train, y_train, A_train = load_and_preprocess(config["dataset"], config["dataset"]["path_train"])
    X_test, y_test, A_test    = load_and_preprocess(config["dataset"], config["dataset"]["path_test"])


    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    #df_train = pd.concat([X_train.reset_index(drop=True), 
    #                  y_train.reset_index(drop=True), 
    #                  A_train.reset_index(drop=True)], axis=1)
    #df_train.to_csv('dftrain_antes_standard.csv', index=False)

    colunas = X_train.columns.tolist()  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=colunas)
    X_test  = pd.DataFrame(X_test, columns=colunas)

    #df_train = pd.concat([X_train.reset_index(drop=True), 
    #                  y_train.reset_index(drop=True), 
    #                  A_train.reset_index(drop=True)], axis=1)
    #df_train.to_csv('dftrain_depois_standard.csv', index=False)
   
    # === Pre-processing mitigation ===

    X_train, y_train, A_train, params_pre_mitigation = apply_mitigation_pre(X_train, y_train, A_train, config["mitigation"]["pre"])

    # === Model training  ===
    model = train_model(
        config["model"]["name"].lower(), 
        X_train, y_train, 
        params_model=config["model"]["params"], 
        params_fit=params_pre_mitigation
    )

    # === In-processing mitigation ===
    model = apply_mitigation_in(
        model, 
        X_train, y_train, A_train, 
        config["mitigation"]["in"]
    )

    # === Test ===
    y_pred  = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred.astype(float)  # transforma 0/1 em float # ruim para roc_auc e outras que usam proba, pq colocamos apenas o pred e não proba

    
    #y_proba = model.predict_proba(X_test)[:, 1]
    
    #if config["mitigation"]["in"]["name"] == 'none': # hasattr(model, "predict_proba"):
    #    y_pred  = model.predict(X_test)
    #    y_proba = model.predict_proba(X_test)[:, 1]
    #    print('nao usei in-process mitigation')
    #else:
    #    print('usei in-process mitigation')
    #    dataset_binary = pd.DataFrame(X_test)
    #    dataset_binary['target'] = y_test.values
    #    dataset_binary['sensitive'] = A_test.values
    #    dataset_test = BinaryLabelDataset(
    #        df=dataset_binary,
    #        label_names=['target'],
    #        protected_attribute_names=['sensitive'],
    #        favorable_label=1,
    #        unfavorable_label=0
    #    )
    #    y_pred = model.predict(dataset_test).labels.ravel()
        #y_proba = y_pred  # AIF360 não retorna probabilidade, apenas labels (ruim para log-loss, auc, roc..). Usar Fairlearn 
                          # Evitar AIF360 se você precisa de probabilidades calibradas — use modelos fairness-aware que integram com sklearn (como fairlearn).
    
    # === Post-processing mitigation ===
    y_pred, y_proba = apply_mitigation_post(y_pred, y_proba, y_test, A_test, config["mitigation"]["post"])

    # === Info Model e Mitigation  ===
    model_info = pd.DataFrame([{
        "id":    str(uuid.uuid4()),
        "data":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_execution": time.time() - start_time ,
        "model": config["model"]["name"].lower()
    }])

    mitigation_info = pd.DataFrame([{
        "pre":  config["mitigation"]["pre"]["name"],
        "in":   config["mitigation"]["in"]["name"],
        "post": config["mitigation"]["post"]["name"]
    }])

    # === Performance Metrics ===
    performance_metrics = evaluate_performance(y_test, y_pred, y_proba)

    # === Fairness Metrics ===
    fairness_metrics = evaluate_fairness(
        y_test, y_pred, A_test,
        sensitive_attribute=config["dataset"]["sensitive"],
        target=config["dataset"]["target"]
    )

    results = pd.concat([model_info, mitigation_info, performance_metrics, fairness_metrics], axis=1)

    # === Save results ===
    results_file = "runs_"+config["dataset"]["name"]+".csv"
    if os.path.exists(results_file):
        results.to_csv(results_file, mode="a", header=False, index=False)
    else:
        results.to_csv(results_file, index=False)

    print("Save:", results_file)
    print(results)

    return results

if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"

    run_experiment(config_path)
