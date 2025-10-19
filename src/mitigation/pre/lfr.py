def apply(X_train, y_train, A_train, params):
    """
    Aplica Learning Fair Representations (LFR) do AIF360.
    
    Args:
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.Series): Labels binárias.
        A_train (pd.Series): Atributo protegido binário.
        params (dict): Parâmetros do LFR, ex:
            {
                "k": 5,
                "Ax": 0.01,
                "Ay": 1.0,
                "Az": 50.0,
                "maxiter": 5000,
                "maxfun": 5000,
                "threshold": 0.5,
                "verbose": 0,
                "seed": 42
            }
    
    Returns:
        X_train_transf, y_train_transf, A_train_transf, params_used
    """
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import LFR

    # === 1. Cria DataFrame combinado ===
    y_train = y_train.rename("label_bin")
    A_train = A_train.rename("protected_bin")

    # Combina tudo em um único DataFrame
    df = pd.concat([X_train, y_train, A_train], axis=1)
    df["label_bin"] = df["label_bin"].replace({None: 0}).fillna(0).astype(float)
    df["protected_bin"] = df["protected_bin"].replace({None: 0}).fillna(0).astype(float)

    print(df)

    invalid_cols = df.select_dtypes(include=['object', 'category', 'bool', 'float']).columns.tolist()
    if invalid_cols:
        print(f"\n Colunas potencialmente inválidas (não numéricas): {invalid_cols}")
    else:
        print("\n Todas as colunas são numéricas (aparentemente seguras para o LFR).")

    # === 2. Cria BinaryLabelDataset ===
    dataset = BinaryLabelDataset(
        df=df,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        favorable_label=1,
        unfavorable_label=0
    )
 
    # === 3. Define grupos privilegiados e não privilegiados ===
    unprivileged_groups = [{"protected_bin": 0}]
    privileged_groups = [{"protected_bin": 1}]

    print("Número de features no dataset:", dataset.features.shape[1])
    print("Colunas no DataFrame original:", df.columns)


    # === 4. Inicializa LFR com os parâmetros fornecidos ===
    lfr = LFR(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        #k=params.get("k"),
        #Ax=params.get("Ax"),
        #Ay=params.get("Ay"),
        #Az=params.get("Az"),
        k=5,
        Ax=0.01,
        Ay=25,
        Az=0.01,
      
        #print_interval=params.get("print_interval"),
        #verbose=params.get("verbose"),
        #seed=params.get("seed")
    )

    # === 5. Aplica fit_transform ===
    dataset_transf = lfr.fit_transform(
        dataset,
        #maxiter=params.get("maxiter"),
        #maxfun=params.get("maxfun"),
        #threshold=params.get("threshold")

        maxiter=100,
        maxfun=100,   
        threshold=0.5     
    )

    # === 6. Reconstrói dataframes ===
    orig_cols_X = list(X_train.columns)
    features = pd.DataFrame(dataset_transf.features, columns=orig_cols_X + ["protected_bin"])
    X_train_transf = features.drop(columns=["protected_bin"])
    y_train_transf = pd.Series(dataset_transf.labels.ravel(), name="label_bin")
    A_train_transf = pd.Series(dataset_transf.protected_attributes.ravel(), name="protected_bin")

    # === 7. Validação ===
    assert len(X_train_transf) == len(y_train_transf) == len(A_train_transf), \
        "Erro: tamanhos não batem após LFR"

    unique_classes = sorted(set(y_train_transf))
    if len(unique_classes) < 2:
        print(f"[LFR ALERT] Only one class found. Try reducing Az  or increasing Ay.")

    return X_train_transf, y_train_transf, A_train_transf, params
