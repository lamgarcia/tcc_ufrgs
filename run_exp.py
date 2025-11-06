import os
import time
import sys
import uuid
import yaml
import importlib

import pandas as pd
from datetime import datetime
from src.metrics.evaluate_performance import evaluate_performance
from src.metrics.evaluate_fairness import evaluate_fairness


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

 
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

    # remove colunas indicadas em 'cols_exclude' (se existirem) ===
    cols_exclude = dataset_cfg.get("cols_exclude", [])
    if cols_exclude:
        # Garante que as colunas realmente existem antes de tentar remover
        cols_present = [c for c in cols_exclude if c in df.columns]
        if cols_present:
            df.drop(columns=cols_present, inplace=True)
        else:
            print(f"Aviso: Nenhuma das colunas em {cols_exclude} foi encontrada no dataset.")

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


    # Retirando caracteres inválidos dos nomes das colunas
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

    # === One-Hot Encoding ===
    cols_cat = dataset_cfg.get("cols_cat", [])
    
    cols_cat_present = [c for c in cols_cat if c in X.columns]

    if cols_cat_present:
        X[cols_cat_present] = X[cols_cat_present].astype('category')
        X = pd.get_dummies(X, columns=cols_cat_present, drop_first=False)
    else:
        print(f"Nenhuma das colunas categóricas {cols_cat} foi encontrada em X.")

    print(f"Shape final de X após one-hot: {X.shape}")
    print(f"Tipos finais:\n{X.dtypes.value_counts()}")
   
    #transformar true/false em 1 e 0
    #X[cols_cat_present] = X[cols_cat_present].astype(int)

    return X, y, A

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

def apply_mitigation_post(X_val, y_val, A_val, y_val_pred, y_val_proba, X_test, y_test, A_test, y_test_pred, y_test_proba, mitigation_cfg):

    if mitigation_cfg["name"].lower() == "none":
        return y_test_pred, y_test_proba
    try:
        module_path = f"src.mitigation.post.{mitigation_cfg['name'].lower()}"
        mitigation_module = importlib.import_module(module_path)
        return mitigation_module.apply(X_val, y_val, A_val, y_val_pred, y_val_proba, X_test, y_test, A_test, y_test_pred, y_test_proba, mitigation_cfg.get("params", {}))
    except ModuleNotFoundError:
            raise ValueError(f"Mitigation method '{mitigation_cfg['name']}' not found in '{module_path}'")
    except AttributeError:
            raise ValueError(f"The module {module_path} must have a function 'apply(X_val, y_val, A_val, y_val_pred, y_val_proba, X_test, y_test, A_test, y_test_pred, y_test_proba, params)'")
    

# ===================== Run Experiment =====================
def run_experiment(config_path):

    start_time = time.time()
    config = load_config(config_path)

    model_name      = config["model"]["name"]
    method_pre_mitigation  = config["mitigation"]["pre"]["name"] 
    method_in_mitigation   = config["mitigation"]["in"]["name"]
    method_post_mitigation = config["mitigation"]["post"]["name"]

    # === Data Preprocess  ===

    X_train, y_train, A_train = load_and_preprocess(config["dataset"], config["dataset"]["path_train"])
    X_val, y_val, A_val      = load_and_preprocess(config["dataset"], config["dataset"]["path_val"])
    X_test, y_test, A_test    = load_and_preprocess(config["dataset"], config["dataset"]["path_test"])

    # Preenche com zeros se houver colunas faltantes depois do pre-processamento
    if (list(X_train.columns) != list(X_test.columns)) or (list(X_train.columns) != list(X_val.columns)):
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
        X_train, X_val = X_train.align(X_val, join="left", axis=1, fill_value=0)      

    #df_train = pd.concat([X_train.reset_index(drop=True),  y_train.reset_index(drop=True), A_train.reset_index(drop=True)], axis=1)
    #df_train = pd.concat([X_train,  y_train, A_train], axis=1)
    #df_train.to_csv('dftrain_depois_preprocessamento_dataset.csv', index=False)
   
    # === Pre-processing mitigation ===

    #DEBUG
    df_train = pd.concat([X_train, y_train, A_train], axis=1)
    file_name = f"{model_name}__pre-{method_pre_mitigation}__in-{method_in_mitigation}__post-{method_post_mitigation}"
    df_train.to_csv(f"dfs/{file_name}_1_entrada_pre_mitigacao_TRAIN.csv", index=False)

    X_train, y_train, A_train, params_pre_mitigation = apply_mitigation_pre(X_train, y_train, A_train, config["mitigation"]["pre"])

    # === Model training  ===
    model = train_model(
        model_name.lower(), 
        X_train, y_train, 
        params_model=config["model"]["params"], 
        params_fit=params_pre_mitigation
    )

    #DEBUG
    df_train = pd.concat([X_train, y_train, A_train], axis=1)
    df_train.to_csv(f"dfs/{file_name}_2_saida_pre_mitigacao.csv", index=False)

    # === In-processing mitigation ===
    model = apply_mitigation_in(
        model, 
        X_train, y_train, A_train, 
        config["mitigation"]["in"]
    )
    
    # === Test ===
    y_val_pred  = model.predict(X_val)
    y_test_pred  = model.predict(X_test)

    # Verifica quais modelos retornam probabilidades
    if hasattr(model, "predict_proba"):
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        print(f"{model.__class__.__name__}: sim, tem predict_proba")
    else:
        y_val_proba = y_test_pred.astype(float)
        y_test_proba = y_test_pred.astype(float)  # transforma 0/1 em float # ruim para roc_auc e outras que usam proba, pq colocamos apenas o pred e não proba
        print(f"{model.__class__.__name__}: não tem predict_proba, usando pred como float")
    
     #DEBUG
    df_train = pd.concat([
        pd.Series(y_test_pred, name="y_test_pred"),
        pd.Series(y_test_proba, name="y_test_proba"),
        pd.Series(y_test, name="y_test"),
        pd.Series(A_test, name="A_test")
    ], axis=1)
    df_train.to_csv(f"dfs/{file_name}_3_entrada_post_mitigacao.csv", index=False)

    # === Post-processing mitigation ===
    y_test_pred, y_test_proba = apply_mitigation_post(X_val, y_val, A_val, y_val_pred, y_val_proba, X_test, y_test, A_test, y_test_pred, y_test_proba, config["mitigation"]["post"])

     #DEBUG
    df_train = pd.concat([
        pd.Series(y_test_pred, name="y_test_pred"),
        pd.Series(y_test_proba, name="y_test_proba"),
        pd.Series(y_test, name="y_test"),
        pd.Series(A_test, name="A_test")
    ], axis=1)
    df_train.to_csv(f"dfs/{file_name}_4_saida_post_mitigacao.csv", index=False)

    # === Info Model e Mitigation  ===
    model_info = pd.DataFrame([{
        "id":    str(uuid.uuid4()),
        "data":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_execution": time.time() - start_time ,
        "model": config["model"]["name"].lower()
    }])

    mitigation_info = pd.DataFrame([{
        "pre":  method_pre_mitigation,
        "in":   method_in_mitigation,
        "post": method_post_mitigation
    }])

    # === Performance Metrics ===
    performance_metrics = evaluate_performance(y_test, y_test_pred, y_test_proba)

    # === Fairness Metrics ===

    fairness_metrics = evaluate_fairness(
        y_test, y_test_pred, A_test,
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
