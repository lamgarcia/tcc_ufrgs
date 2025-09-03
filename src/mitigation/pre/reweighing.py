def apply(X_train, y_train, A_train, params):
    import numpy as np
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing

    # Converte tudo para float
    df = pd.DataFrame(X_train.astype(float))
    df['label'] = y_train.astype(float)
    df['protected'] = A_train.astype(float)

    # Remove NaNs ou infinitos
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )

    # Reweighing
    rw = Reweighing(
        unprivileged_groups=[{'protected': 0}],
        privileged_groups=[{'protected': 1}]
    )
    rw.fit(dataset)
    dataset_transf = rw.transform(dataset)

    X_train_new = df.drop(['label', 'protected'], axis=1).values
    y_train_new = df['label'].values
    sample_weight = dataset_transf.instance_weights

    return X_train_new, y_train_new, sample_weight
