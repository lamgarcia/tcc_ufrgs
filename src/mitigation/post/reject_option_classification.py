def apply(
    X_val, y_val, A_val, y_val_pred, y_val_proba,
    X_test, y_test, A_test, y_test_pred, y_test_proba,
    params
):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import RejectOptionClassification

    # ===============================
    # 1️⃣ Cria datasets BinaryLabelDataset de VALIDAÇÃO
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
    # PARA CORREÇÃO 
    val_pred.scores = y_val_proba.reshape(-1, 1)

    # ===============================
    # 2️⃣ Cria datasets BinaryLabelDataset de TESTE
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

    # PARA CORREÇÃO 
    test_pred.scores = y_test_proba.reshape(-1, 1)
    
    # ===============================
    # 3️⃣ Ajusta Reject Option Classification no conjunto de validação
    # ===============================
    roc = RejectOptionClassification(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}],
        **params  # aceita parâmetros extras, ex: threshold, margin, etc.
    )

    roc = roc.fit(val_true, val_pred)

    # ===============================
    # 4️⃣ Aplica o pós-processamento ao conjunto de teste
    # ===============================
    test_roc = roc.predict(test_pred)

    # ===============================
    # 5️⃣ Extrai as predições ajustadas
    # ===============================
    y_test_pred_roc = test_roc.labels.ravel()

    # ROC também não altera probabilidades — mantém as originais
    y_test_proba_roc = y_test_proba

    return y_test_pred_roc, y_test_proba_roc
