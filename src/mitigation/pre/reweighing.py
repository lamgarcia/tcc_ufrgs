def apply(X_train, y_train, A_train, params):

    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    import pandas as pd

    df = pd.concat([X_train, y_train, A_train], axis=1)

    # Cria BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label_bin'],
        protected_attribute_names=['protected_bin'],
        favorable_label=1,
        unfavorable_label=0
    )
    # Aplica Reweighting
    rw = Reweighing(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}]
    )
    #rw.fit(dataset)
    dataset_transf =  rw.fit_transform(dataset)

    # Obtém os pesos
    sample_weight = dataset_transf.instance_weights

    # Validação
    assert len(X_train) == len(y_train)  == len(A_train) == len(sample_weight), \
        "Erro: tamanhos não batem após Reweighting"

    return X_train, y_train, A_train, {"sample_weight": sample_weight}