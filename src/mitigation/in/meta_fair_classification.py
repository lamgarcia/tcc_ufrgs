def apply(model, X_train, y_train, A_train, params):
    """
    Aplica mitigação in-processing usando MetaFairClassifier
    mantendo o modelo original como estimador base.

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

    from aif360.algorithms.inprocessing import MetaFairClassifier
    import numpy as np

    # Define a métrica de fairness para MetaFairClassifier
    constraint_type = params.get('type')
    
    if constraint_type not in ['fdr', 'sr']:
        raise ValueError(f"Constraint {constraint_type} not supported.")

    # MetaFairClassifier usa tau como trade-off fairness/accuracy
    tau = params.get('tau')

    # Garante que A_train seja array numpy
    A_train = np.array(A_train)

    # Cria o MetaFairClassifier usando o modelo base
    mitigator = MetaFairClassifier(
        tau=tau,
        sensitive_attr=A_train,
        base_model=model,
        type=constraint_type
    )

    # Ajusta o mitigador
    mitigator.fit(X_train, y_train)

    return mitigator
