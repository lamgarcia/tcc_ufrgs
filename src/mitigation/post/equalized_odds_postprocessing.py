def apply(y_pred, y_proba, y_test, A_test, params):

    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import EqOddsPostprocessing

    # Monta DataFrame para AIF360
    df = pd.DataFrame({
        "label_bin": y_test,
        "pred": y_pred,
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

    # Dataset predito
    df_pred = df.copy()
    df_pred["label_bin"] = y_pred
    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # Instancia e ajusta Equalized Odds
    eq_odds = EqOddsPostprocessing(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}]
    )

    eq_odds = eq_odds.fit(dataset_true, dataset_pred)

    # Transforma as previsões
    dataset_pred_transf = eq_odds.predict(dataset_pred)

    # Extrai previsões corrigidas
    y_pred_transf = dataset_pred_transf.labels.ravel()
 

    return y_pred_transf, y_proba
