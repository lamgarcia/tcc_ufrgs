def apply_dir(X_train, y_train, A_train, repair_level=1.0):
    """
    Aplica DisparateImpactRemover do AIF360 como técnica de pré-mitigation.
    Retorna os dados limpos e as features transformadas no dicionário de atributos.

    Parâmetros:
        X_train: features de treino (array ou DataFrame)
        y_train: rótulos (0/1)
        A_train: atributo sensível (0/1)
        repair_level: nível de reparo (0.0 = nenhum, 1.0 = máximo)

    Retorna:
        X_clean: features originais (após limpeza)
        y_clean: rótulos limpos
        A_clean: atributo sensível limpo
        dict: contendo {'X_transf': array das features transformadas}
    """
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    import pandas as pd
    import numpy as np

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

    # Aplica DisparateImpactRemover
    dir_algorithm = DisparateImpactRemover(
        sensitive_attribute='protected',
        repair_level=repair_level  # 0.0 = original, 1.0 = máximo de reparo
    )

    dataset_transf = dir_algorithm.fit_transform(dataset)

    # Extrai as features transformadas (mantém label e protected intactos)
    # As últimas 2 colunas são 'label' e 'protected' → removemos elas
    X_transf = dataset_transf.features[:, :-2]

    # Validação
    assert len(X_clean) == len(X_transf) == len(y_clean) == len(A_clean), \
        "Erro: tamanhos não batem após DisparateImpactRemover"

    # RETORNO NO FORMATO ESPERADO:
    return X_clean, y_clean, A_clean, {"X_transf": X_transf}