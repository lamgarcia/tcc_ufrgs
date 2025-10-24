def apply(model, X_train, y_train, A_train, params):
    """
    Aplica mitigação in-processing inspirada no MetaFairClassifier,
    mantendo o modelo original como estimador base (agnóstico).

    model : objeto sklearn-like
        Modelo não treinado (RandomForest, XGBClassifier, LogisticRegression etc.)
    X_train : DataFrame ou array-like
        Features de treino.
    y_train : array-like
        Labels binários de treino.
    A_train : array-like
        Atributo protegido binário de treino.
    params : dict
        Parâmetros da técnica, ex: {"type": "sr" ou "fdr", "tau": 0.8}
    """

    import numpy as np
    import inspect

    # --- Validação dos parâmetros ---
    fairness_type = params.get("type")   # 'sr' ou 'fdr'
    tau = float(params.get("tau"))

    if fairness_type not in ["sr", "fdr"]:
        raise ValueError(f"Fairness type '{fairness_type}' not supported. Use 'sr' or 'fdr'.")

    # --- Converte para numpy arrays ---
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    A_train = np.asarray(A_train)

    # --- Valida compatibilidade com sample_weight ---
    if "sample_weight" not in inspect.signature(model.fit).parameters:
        class Wrapper(model.__class__):
            def fit(self, X, y, sample_weight=None, **kwargs):
                return super().fit(X, y, **kwargs)
        model = Wrapper(**model.get_params())

    # --- Funções auxiliares de fairness ---
    def statistical_rate(y_pred, A):
        """SR = P(y_hat=1 | A=0) / P(y_hat=1 | A=1)"""
        pos_unpriv = np.mean(y_pred[A == 0])
        pos_priv = np.mean(y_pred[A == 1])
        return pos_unpriv / (pos_priv + 1e-8)

    def fdr_ratio(y_pred, y_true, A):
        """FDR ratio = (FP_rate_unpriv) / (FP_rate_priv)"""
        fp_unpriv = np.mean((y_pred[A == 0] == 1) & (y_true[A == 0] == 0))
        fp_priv = np.mean((y_pred[A == 1] == 1) & (y_true[A == 1] == 0))
        return fp_unpriv / (fp_priv + 1e-8)

    # --- Treinamento iterativo com fairness weighting ---
    sample_weight = np.ones(len(y_train))

    for epoch in range(20):  # pode ajustar número de iterações
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_train)

        # calcula a métrica de fairness atual
        if fairness_type == "sr":
            fairness_value = statistical_rate(y_pred, A_train)
            fairness_penalty = np.abs(1 - fairness_value)  # quanto mais distante de 1, pior
        else:  # fdr
            fairness_value = fdr_ratio(y_pred, y_train, A_train)
            fairness_penalty = np.abs(1 - fairness_value)

        # ajusta pesos para reduzir disparidade
        # instâncias de grupos desprivilegiados ganham mais peso
        scale = 10.0  # fator de amplificação da fairness
        group_weights = np.where(A_train == 0,
                                np.exp(tau * scale * fairness_penalty),
                                np.exp(-tau * scale * fairness_penalty))
        group_weights = np.clip(group_weights, 0.1, 20)

        #group_weights = np.where(A_train == 0, 1 + tau * fairness_penalty, 1 - tau * fairness_penalty)
        #group_weights = np.clip(group_weights, 0.1, 10)  # estabilidade numérica

        sample_weight = group_weights




    return model
