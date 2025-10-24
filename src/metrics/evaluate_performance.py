import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    log_loss,
    cohen_kappa_score,
    fbeta_score,
    jaccard_score,
    hamming_loss,
    confusion_matrix
)

def evaluate_performance(y_true, y_pred, y_proba=None):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "precision": precision_score(y_true, y_pred),
        "npv": precision_score(y_true, y_pred, pos_label=0),
        "f1-score": f1_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0,
        "pr_auc": average_precision_score(y_true, y_proba)  if y_proba is not None else 0.0,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba)  if y_proba is not None else 0.0,
        "cohen_kappa_score": cohen_kappa_score(y_true, y_pred),
        "fbeta_score": fbeta_score(y_true, y_pred, beta=2),
        "jaccard_score": jaccard_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred)
    }

    return pd.DataFrame([metrics])
