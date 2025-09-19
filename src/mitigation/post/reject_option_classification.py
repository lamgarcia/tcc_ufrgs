def apply(y_pred, y_proba, y_test, A_test, params):
    """
    Aplica RejectOptionClassification nos resultados preditos.

    y_pred : array-like
        Previsões binárias originais do modelo.
    y_proba : array-like
        Probabilidades preditas pelo modelo.
    y_test : array-like
        Rótulos verdadeiros do conjunto de teste.
    A_test : array-like
        Atributo protegido (binário) do conjunto de teste.
    params : dict
        Parâmetros da técnica (ex: seed, grupos privilegiados e desprivilegiados).
    """

    import pandas as pd
    import numpy as np
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import RejectOptionClassification

    # Default dos grupos (pode sobrescrever via params)
    unprivileged_groups = params.get("unprivileged_groups", [{'protected_bin': 0}])
    privileged_groups   = params.get("privileged_groups", [{'protected_bin': 1}])

    # Monta DataFrame para AIF360
    df = pd.DataFrame({
        "label_bin": y_test,
        "score": y_proba,       # ROC usa score (probabilidade)
        "protected_bin": A_test
    })

    # Dataset verdadeiro
    dataset_true = BinaryLabelDataset(
        df=df,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # Dataset predito (com score)
    df_pred = df.copy()
    df_pred["label_bin"] = y_pred
    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0,
    )

    dataset_pred.scores = np.array(y_proba).reshape(-1, 1)


    # Instancia Reject Option Classification
    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        metric_name=params.get("metric_name", "Statistical parity difference"),
        metric_ub=params.get("metric_ub", 0.05),
        metric_lb=params.get("metric_lb", -0.05)        
        )

    roc = roc.fit(dataset_true, dataset_pred)

    # Transforma as previsões
    dataset_pred_transf = roc.predict(dataset_pred)

    # Extrai previsões corrigidas
    y_pred_transf = dataset_pred_transf.labels.ravel()

    print(y_pred_transf)
    print(y_proba)

    return y_pred_transf, y_proba
