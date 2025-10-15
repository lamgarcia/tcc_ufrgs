def apply(y_pred, y_proba, y_test, A_test, params):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import DeterministicReranking

    df = pd.DataFrame({
        "label_bin": y_test,
        "protected_bin": A_test,
        "score": y_proba
    })
    df_pred = df.copy()
    df_pred["label_bin"] = y_pred

    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        scores_names=["score"],
        favorable_label=1,
        unfavorable_label=0
    )

    reranker = DeterministicReranking(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}]
    )

    # === Parâmetros ajustados ===
    rec_size = int(params.get("rec_size", 0.3 * len(df)))
    rerank_type = params.get("rerank_type")
    renormalize = params.get("renormalize_scores", True)
    group_counts = df["protected_bin"].value_counts(normalize=True).to_dict()
    target_prop = params.get("target_prop", [group_counts.get(0, 0.5), group_counts.get(1, 0.5)])

    # === Executa reranking ===
    dataset_reranked = reranker.fit_predict(
        dataset=dataset_pred,
        rec_size=rec_size,
        target_prop=target_prop,
        rerank_type=rerank_type,
        renormalize_scores=renormalize
    )

    # === Extrai previsões reranqueadas ===
    y_pred_rerank = dataset_reranked.labels.ravel()
    y_proba_rerank = dataset_reranked.scores.ravel()

    # === Reconstrói vetor completo (completando o resto com originais) ===
    y_pred_transf = y_pred.copy()
    y_proba_transf = y_proba.copy()

    # substitui as rec_size primeiras posições (ou usa índice retornado se disponível)
    n_rerank = min(rec_size, len(y_pred_rerank))
    y_pred_transf[:n_rerank] = y_pred_rerank
    y_proba_transf[:n_rerank] = y_proba_rerank

    return y_pred_transf, y_proba_transf
