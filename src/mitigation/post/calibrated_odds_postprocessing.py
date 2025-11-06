def apply(
    X_val, y_val, A_val, y_val_pred, y_val_proba,
    X_test, y_test, A_test, y_test_pred, y_test_proba,
    params
):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

    # ===============================
    # 1️⃣ Cria datasets BinaryLabelDataset de VALIDAÇÃO
    # ===============================
    df_val = pd.DataFrame({
        "label_bin": y_val,
        "protected_bin": A_val,
        "score": y_val_proba,  # probabilidade para calibrar
    })
    
    print(y_val_proba)
    
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
    # 2️⃣ Cria datasets BinaryLabelDataset de TESTE
    # ===============================
    df_test = pd.DataFrame({
        "label_bin": y_test,
        "protected_bin": A_test,
        "score": y_test_proba,  # probabilidade original
    })

    test_pred = BinaryLabelDataset(
        df=df_test.assign(label_bin=y_test_pred),
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # ===============================
    # 3️⃣ Ajusta Calibrated Equalized Odds no conjunto de validação
    # ===============================
    ceo = CalibratedEqOddsPostprocessing(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}],
        **params  # aceita 'cost_constraint', 'seed', etc.
    )

    ceo = ceo.fit(val_true, val_pred)

    # ===============================
    # 4️⃣ Aplica o pós-processamento ao conjunto de teste
    # ===============================
    test_ceo = ceo.predict(test_pred)

    # ===============================
    # 5️⃣ Extrai as predições ajustadas
    # ===============================
    y_test_pred_eq = test_ceo.labels.ravel()
    y_test_proba_eq = test_ceo.scores.ravel()  # calibradas

    return y_test_pred_eq, y_test_proba_eq
