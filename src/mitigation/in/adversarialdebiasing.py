# Exemplo simplificado usando AIF360
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

def apply(model, X_train, y_train, A_train, params):
    """
    Aplica Adversarial Debiasing (in-processing).
    Retorna um modelo já treinado com fairness in-processing.
    """
    # Converte para BinaryLabelDataset
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset

    df = pd.DataFrame(X_train)
    df['label'] = y_train
    df['protected'] = A_train

    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )

    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    adv_model = AdversarialDebiasing(
        privileged_groups=[{'protected': 1}],
        unprivileged_groups=[{'protected': 0}],
        scope_name='adv_debiasing',
        sess=sess,
        num_epochs=params.get("num_epochs", 50),
        batch_size=params.get("batch_size", 128),
        learning_rate=params.get("learning_rate", 0.01)
    )
    adv_model.fit(dataset)

    return adv_model
