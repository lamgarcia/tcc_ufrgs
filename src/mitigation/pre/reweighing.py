def apply(X_train, y_train, A_train, params):

    from aif360.algorithms.preprocessing import Reweighing

    # Converte para DataFrame/Series com índice alinhado
    X_train_df = pd.DataFrame(X_train.astype(float))
    y_train_sr = pd.Series(y_train.astype(float), name='label')
    A_train_sr = pd.Series(A_train.astype(float), name='protected')

    # Combina tudo
    df = pd.concat([X_train_df, y_train_sr, A_train_sr], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Extraímos os dados limpos
    X_clean = df.drop(['label', 'protected'], axis=1).values
    y_clean = df['label'].values
    A_clean = df['protected'].values

    # Cria BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )

    # Aplica Reweighting
    rw = Reweighing(
        unprivileged_groups=[{'protected': 0}],
        privileged_groups=[{'protected': 1}]
    )
    #rw.fit(dataset)
    dataset_transf =  rw.fit_transform(dataset)

    # Obtém os pesos
    sample_weight = dataset_transf.instance_weights

    # Validação
    assert len(X_clean) == len(y_clean) == len(sample_weight), \
        "Erro: tamanhos não batem após Reweighting"

    return X_clean, y_clean, A_clean, sample_weight
