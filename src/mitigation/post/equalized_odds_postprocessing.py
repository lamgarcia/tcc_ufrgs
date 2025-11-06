def apply(
    X_val, y_val, A_val, y_val_pred, y_val_proba,
    X_test, y_test, A_test, y_test_pred, y_test_proba,
    params
):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import EqOddsPostprocessing

    # ===============================
    # 1Cria datasets BinaryLabelDataset de VALIDAÇÃO
    # ===============================
    df_val = pd.DataFrame({
        "label_bin": y_val,
        "protected_bin": A_val,
        "pred": y_val_pred
    })

    val_true = BinaryLabelDataset(
        df=df_val,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    val_pred = BinaryLabelDataset(
        df=df_val.assign(label_bin=y_val_pred),
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # ===============================
    # Cria datasets BinaryLabelDataset de TESTE
    # ===============================
    df_test = pd.DataFrame({
        "label_bin": y_test,
        "protected_bin": A_test,
        "pred": y_test_pred
    })

    test_pred = BinaryLabelDataset(
        df=df_test.assign(label_bin=y_test_pred),
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # ===============================
    # 3️Ajusta Equalized Odds no conjunto de validação
    # ===============================
    eq_odds = EqOddsPostprocessing(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}],
        **params  # permite passar parâmetros extras
    )

    eq_odds = eq_odds.fit(val_true, val_pred)

    # ===============================
    #  Aplica o pós-processamento ao conjunto de teste
    # ===============================
    test_eq = eq_odds.predict(test_pred)

    # ===============================
    #  Extrai as predições ajustadas
    # ===============================
    y_test_pred_eq = test_eq.labels.ravel()

    # EqOdds não altera probabilidades — retorna as originais
    y_test_proba_eq = y_test_proba

    return y_test_pred_eq, y_test_proba_eq
