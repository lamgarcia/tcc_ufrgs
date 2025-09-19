def apply(model, X_train, y_train, A_train, params):

    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    import inspect
    
    # Seleciona constraint
    constraint_type =  params['constraint']
    print ("constraint: ",constraint_type )

    if constraint_type == "DemographicParity":
        constraint = DemographicParity()
    elif constraint_type == "EqualizedOdds":
        constraint = EqualizedOdds()
    else:
        raise ValueError(f"Constraint {constraint_type} not supported.")
    
    # --- valida compatibilidade do modelo com o paramentor sample_weight ---
    if "sample_weight" not in inspect.signature(model.fit).parameters:
        # Wrapper simples que ignora sample_weight
        class Wrapper(model.__class__):
            def fit(self, X, y, sample_weight=None, **kwargs):
                return super().fit(X, y, **kwargs)
        model = Wrapper(**model.get_params())

    # Cria mitigador usando o modelo original como estimador base
    mitigator = ExponentiatedGradient(
        estimator=model,
        constraints=constraint,
        eps=params['eps']
    )

    # Ajusta o mitigador (pode re-treinar pequenas variações)
    mitigator.fit(X_train, y_train, sensitive_features=A_train)

    # Retorna somente o objeto mitigador
    return mitigator



