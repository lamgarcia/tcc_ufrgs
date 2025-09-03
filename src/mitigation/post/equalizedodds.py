from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
import pandas as pd

def apply(y_pred, y_proba, y_val, A_val, params):
    """
    Aplica Equalized Odds Post-processing.
    Ajusta predições para reduzir diferença de eq. de odds.
    """
    df_true = pd.DataFrame({
        "label": y_val.values,
        "protected": A_val.values
    })
    dataset_true = BinaryLabelDataset(
        df=df_true,
        label_names=['label'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )

    df_pred = pd.DataFrame({
        "label": y_pred,
        "protected": A_val.values
    })
    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=['label'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )

    eqodds = EqOddsPostprocessing(
        privileged_groups=[{'protected': 1}],
        unprivileged_groups=[{'protected': 0}]
    )
    eqodds = eqodds.fit(dataset_true, dataset_pred)
    dataset_pred_transf = eqodds.predict(dataset_pred)

    y_pred_new = dataset_pred_transf.labels.ravel()
    y_proba_new = y_proba  # Não altera probabilidades, só labels

    return y_pred_new, y_proba_new
