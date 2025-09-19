def apply(model, X_train, y_train, A_train, mitigation_cfg):
    """
    Aplica mitigação in-processing usando Fairlearn GridSearchReduction
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
        Parâmetros da técnica, ex: {"constraint": "DemographicParity", "grid_size": 10}
    """

    from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
    import inspect

    # Seleciona constraint
    constraint_type = mitigation_cfg.get("constraint", "DemographicParity")
    if constraint_type == "DemographicParity":
        constraint = DemographicParity()
    elif constraint_type == "EqualizedOdds":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Constraint {constraint_type} não suportada.")
    
    # --- valida compatibilidade do modelo com o paramentor sample_weight do grid search---
    if "sample_weight" not in inspect.signature(model.fit).parameters:
        # Wrapper simples que ignora sample_weight
        class Wrapper(model.__class__):
            def fit(self, X, y, sample_weight=None, **kwargs):
                return super().fit(X, y, **kwargs)
        model = Wrapper(**model.get_params())

    # Cria mitigador usando o modelo original como estimador base
    mitigator = GridSearch(
        estimator=model,
        constraints=constraint,
        grid_size=mitigation_cfg.get("grid_size", 3)
    )

    # Ajusta o mitigador
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    # Retorna somente o objeto mitigador
    return mitigator
