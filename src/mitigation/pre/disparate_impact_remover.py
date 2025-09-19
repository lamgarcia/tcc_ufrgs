def apply(X_train, y_train, A_train, params):

    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    import pandas as pd

    df = pd.concat([X_train, y_train, A_train], axis=1)

    # Guarda os nomes originais das features
    orig_cols_X = list(X_train.columns)

    # Cria BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label_bin'],
        protected_attribute_names=['protected_bin'],
        favorable_label=1,
        unfavorable_label=0
    )

    # Aplica Disparate Impact Remover
    di = DisparateImpactRemover(repair_level=params['repair_level']) 
    dataset = di.fit_transform(dataset)

    features = pd.DataFrame(dataset.features, columns=orig_cols_X + ['protected_bin']) # retorna com o protected, tem que retirar para manter padrão
    X_train  = features.drop(features.columns[-1], axis=1)  # todas menos a última
    y_train = pd.Series(dataset.labels.ravel(), name='label_bin')
    A_train = pd.Series(dataset.protected_attributes.ravel(), name='protected_bin')

    # Validação
    assert len(X_train) == len(y_train) == len(A_train), \
        "Erro: tamanhos não batem após Disparate Impact Remover"

    return X_train, y_train, A_train, {}
