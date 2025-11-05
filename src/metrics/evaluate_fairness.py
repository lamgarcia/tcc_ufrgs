import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from fairlearn.metrics import equalized_odds_ratio, equal_opportunity_ratio, demographic_parity_ratio

def evaluate_fairness(y_true, y_pred, A, sensitive_attribute, target):

    dataset_true = BinaryLabelDataset(
        df=pd.DataFrame({target: y_true, sensitive_attribute: A}),
        label_names=[target],
        protected_attribute_names=[sensitive_attribute],
        favorable_label=1,
        unfavorable_label=0
    )

    #dataset_pred = dataset_true.copy()
    #dataset_pred.labels = y_pred
    dataset_pred = dataset_true.copy()
    dataset_pred.labels = pd.DataFrame(y_pred).astype(float).values


    #print ("------------------------------DATASET TRUE --------------------------")
    #print("Features:", dataset_true.features)   # colunas consideradas como features
    #print("Labels:", dataset_true.labels)       # array de labels
    #print("Protected attributes:", dataset_true.protected_attributes)  # atributo protegido
    #print("Feature names:", dataset_true.feature_names)             # features
    #print("Label name:", dataset_true.label_names)                  # target
    #print("Protected attribute names:", dataset_true.protected_attribute_names)  # sensível

    
    #print ("------------------------------DATASET PRED --------------------------")
    #print("Features:", dataset_pred.features)   # colunas consideradas como features
    #print("Labels:", dataset_pred.labels)       # array de labels
    #print("Protected attributes:", dataset_pred.protected_attributes)  # atributo protegido
    #print("Feature names:", dataset_pred.feature_names)             # features
    #print("Label name:", dataset_pred.label_names)                  # target
    #print("Protected attribute names:", dataset_pred.protected_attribute_names)  # sensível


    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        privileged_groups=[{sensitive_attribute: 1}],
        unprivileged_groups=[{sensitive_attribute: 0}]
    )

    pg_value = True  # True for privileged_groups
    ug_value = False # False para unprivileged_groups

    fpr_priv = metric.false_positive_rate(privileged=pg_value)
    fpr_unpriv = metric.false_positive_rate(privileged=ug_value)
    fpr_ratio = (fpr_priv / fpr_unpriv) if fpr_unpriv > 0 else None

    fnr_priv = metric.false_negative_rate(privileged=pg_value)
    fnr_unpriv =  metric.false_negative_rate(privileged=ug_value)
    fnr_ratio = fnr_priv / fnr_unpriv if fnr_unpriv > 0 else None    

    tpr_priv = metric.recall(privileged=pg_value)
    tpr_unpriv = metric.recall(privileged=ug_value)

    tnr_priv =  metric.specificity(privileged=pg_value)
    tnr_unpriv =  metric.specificity(privileged=ug_value)
    tnr_ratio = (tnr_priv / tnr_unpriv) if tnr_unpriv > 0 else None

    npv_priv = metric.negative_predictive_value(privileged=pg_value)
    npv_unpriv = metric.negative_predictive_value(privileged=ug_value)
    npv_ratio =  (npv_priv / npv_unpriv) if npv_unpriv > 0 else None  

    ppv_priv = metric.positive_predictive_value(privileged=pg_value) # precision
    ppv_unpriv = metric.positive_predictive_value(privileged=ug_value)


    if ppv_priv == 0 or ppv_unpriv == 0:
        predictive_parity_ratio = 0.0
    else:
        predictive_parity_ratio = min(ppv_priv / ppv_unpriv, ppv_unpriv / ppv_priv)

    fairlearn_args = {
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive_features": A,
        "method": "between_groups"
    }
    
    metrics = {
        #ratios
        "demographic_parity_ratio": demographic_parity_ratio(**fairlearn_args),       #metric.disparate_impact()
        "equal_odds_ratio": equalized_odds_ratio(**fairlearn_args, agg='worst_case'),                                                   
        "predictive_parity_ratio":  predictive_parity_ratio,
        "equal_opportunity_ratio": equal_opportunity_ratio(**fairlearn_args),
        "tnr_ratio": tnr_ratio,
        "fpr_ratio": fpr_ratio,
        "npv_ratio": npv_ratio,
        "fnr_ratio": fnr_ratio,
        "error_rate_ratio": metric.error_rate_ratio(),
        "average_odds_difference": metric.average_odds_difference(),
        "average_predictive_value_difference": metric.average_predictive_value_difference(),
        "false_positive_rate_ratio": metric.false_positive_rate_ratio(),        
        "false_discovery_rate_ratio": metric.false_discovery_rate_ratio(),
        "false_negative_rate_ratio": metric.false_negative_rate_ratio(),
        "false_omission_rate_ratio": metric.false_omission_rate_ratio(),

        # difference
        "statistical_parity_difference": metric.statistical_parity_difference(),
        "equalized_odds_difference": metric.equalized_odds_difference(),
        "equal_opportunity_difference": metric.equal_opportunity_difference(),
        "error_rate_difference": metric.error_rate_difference(),
        "false_positive_rate_difference": metric.false_positive_rate_difference(),
        "false_discovery_rate_difference": metric.false_discovery_rate_difference(),
        "false_negative_rate_difference": metric.false_negative_rate_difference(),
        "false_omission_rate_difference": metric.false_omission_rate_difference(),
        "true_positive_rate_difference": metric.true_positive_rate_difference(),

        # between groups
        "between_all_groups_coefficient_of_variation": metric.between_all_groups_coefficient_of_variation(),
        "between_all_groups_generalized_entropy_index": metric.between_all_groups_generalized_entropy_index(alpha=2),
        "between_group_generalized_entropy_index": metric.between_group_generalized_entropy_index(alpha=2),
        "between_all_groups_theil_index": metric.between_all_groups_theil_index(),
        "between_group_coefficient_of_variation": metric.between_group_coefficient_of_variation(),
        "between_group_theil_index": metric.between_group_theil_index(),  
        "generalized_entropy_index": metric.generalized_entropy_index(),
        "differential_fairness_bias_amplification": metric.differential_fairness_bias_amplification(concentration=1.0),
        
        # privileged x unprivileged
        "tp_privileged": metric.num_true_positives(privileged=pg_value),
        "tp_unprivileged": metric.num_true_positives(privileged=ug_value),
        "tn_privileged": metric.num_true_negatives(privileged=pg_value),
        "tn_unprivileged": metric.num_true_negatives(privileged=ug_value),
        "fn_privileged": metric.num_false_negatives(privileged=pg_value),
        "fn_unprivileged": metric.num_false_negatives(privileged=ug_value),
        "fp_privileged": metric.num_false_positives(privileged=pg_value),
        "fp_unprivileged": metric.num_false_positives(privileged=ug_value),                        
        "accuracy_privileged": metric.accuracy(privileged=pg_value),
        "accuracy_unprivileged": metric.accuracy(privileged=ug_value),
        "recall_privileged":        tpr_priv,
        "recall_unprivileged":      tpr_unpriv,
        "specificity_privileged":   tnr_priv,
        "specificity_unprivileged": tnr_unpriv,    
        "precision_privileged":     ppv_priv,
        "precision_unprivileged":   ppv_unpriv,
        "npv_privileged": npv_priv,
        "npv_unprivileged": npv_unpriv,

        "selection_rate_privileged": metric.selection_rate(privileged=pg_value),
        "selection_rate_unprivileged": metric.selection_rate(privileged=ug_value),
        "false_positive_rate_privileged": fpr_priv,
        "false_positive_rate_unprivileged": fpr_unpriv,
        "false_negative_rate_privileged": fnr_priv,
        "false_negative_rate_unprivileged": fnr_unpriv,
        "false_discovery_rate_privileged": metric.false_discovery_rate(privileged=pg_value),
        "false_discovery_rate_uprivileged": metric.false_discovery_rate(privileged=ug_value),
        "false_omission_rate_privileged": metric.false_omission_rate(privileged=pg_value),
        "false_omission_rate_uprivileged": metric.false_omission_rate(privileged=ug_value),        
        "error_rate_privileged": metric.error_rate(privileged=pg_value),
        "error_rate_unprivileged": metric.error_rate(privileged=ug_value)
    }

    return pd.DataFrame([metrics])

