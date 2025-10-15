def apply(X_train, y_train, A_train, params):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import OptimPreproc
    from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools

    # === 1. Cria DataFrame combinado ===
    df = pd.concat([X_train, y_train, A_train], axis=1)
    print (df)

    # === 2. Cria BinaryLabelDataset ===
    dataset = BinaryLabelDataset(
        df=df,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )

    # === 3. Define função de distorção genérica ===
    def get_distortion_generic(v_old, v_new):
        """Distância simples: 0 se igual, 1 se diferente."""
        return 0.0 if v_old == v_new else 1.0

    # === 4. Define parâmetros de otimização ===
    optim_options = {
        "distortion_fun": get_distortion_generic,
        "epsilon": params.get("epsilon", 0.05),
        "clist": params.get("clist", [0.99, 1.99, 2.99]),
        "dlist": params.get("dlist", [0.1, 0.05, 0])
    }

    # === 5. Define grupos privilegiados e não privilegiados ===
    unprivileged_groups = [{"protected_bin": 0}]
    privileged_groups = [{"protected_bin": 1}]

    # === 6. Instancia e aplica o otimizador ===
    OP = OptimPreproc(
        optimizer=OptTools,
        optim_options=optim_options,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        verbose=True
    )

    dataset_transf = OP.fit_transform(dataset, transform_Y=True)

    # === 7. Reconstrói DataFrames ===
    orig_cols_X = list(X_train.columns)
    features = pd.DataFrame(dataset_transf.features, columns=orig_cols_X + ["protected_bin"])

    X_train = features.drop(columns=["protected_bin"])
    y_train = pd.Series(dataset_transf.labels.ravel(), name="label_bin")
    A_train = pd.Series(dataset_transf.protected_attributes.ravel(), name="protected_bin")

    # === 8. Validação ===
    assert len(X_train) == len(y_train) == len(A_train), \
        "Erro: tamanhos não batem após OptimPreproc"

    return X_train, y_train, A_train, {"distortion_fun": "generic"}
