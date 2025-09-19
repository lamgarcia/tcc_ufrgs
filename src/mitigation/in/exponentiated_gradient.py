def apply(model, X_train, y_train, A_train, mitigation_cfg):

    """
    Aplica mitigação in-processing usando Fairlearn ExponentiatedGradient
    mantendo o modelo original como estimador base.

    model : objeto sklearn-like
        Modelo já treinado (RandomForest, XGBClassifier, LogisticRegression etc.)
    X_train : DataFrame ou array-like
        Features de treino.
    y_train : array-like
        Labels binários de treino.
    A_train : array-like
        Atributo protegido binário de treino.
    mitigation_cfg : dict
        Parâmetros da técnica, ex: {"constraint": "DemographicParity", "eps": 0.01}
    """

    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds

    # Seleciona constraint
    constraint_type = mitigation_cfg.get("constraint", "DemographicParity")
    if constraint_type == "DemographicParity":
        constraint = DemographicParity()
    elif constraint_type == "EqualizedOdds":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Constraint {constraint_type} não suportada.")

    # Cria mitigador usando o modelo original como estimador base
    mitigator = ExponentiatedGradient(
        estimator=model,
        constraints=constraint,
        eps=mitigation_cfg.get("eps", 0.01)
    )

    # Ajusta o mitigador (pode re-treinar pequenas variações)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    # Retorna somente o objeto mitigador
    return mitigator



