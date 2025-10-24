def apply(y_pred, y_proba, y_test, A_test, params):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

    # Monta DataFrame para AIF360
    df = pd.DataFrame({
        "label_bin": y_test,
        "protected_bin": A_test,
        "score": y_proba          # probabilidade predita usada pelo CalibratedEqOdds
    })

    print(y_proba)

    #df = df.reset_index(drop=True)

    df_pred = df.copy()
    df_pred["label_bin"] = y_pred
    
    #print (df)
    #print(df_pred)

    import matplotlib.pyplot as plt

    plt.hist(y_proba, bins=50)
    plt.title("Distribuição de y_proba")
    plt.show()

    
    print("Distribuição geral:")
    print(df["label_bin"].value_counts())
    print(df["protected_bin"].value_counts())

    print("\nCruzamento label x grupo protegido:")
    print(pd.crosstab(df["label_bin"], df["protected_bin"]))

    print("\nCruzamento previsões x grupo protegido:")
    print(pd.crosstab(df_pred["label_bin"], df_pred["protected_bin"]))

    # Dataset verdadeiro (rótulos reais)
    dataset_true = BinaryLabelDataset(
        df=df,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        scores_names=["score"],    # <-- necessário para o CalibratedEqOdds
        favorable_label=1,
        unfavorable_label=0
    )

    # Dataset predito (com scores e rótulos previstos)
 
    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        label_names=["label_bin"],
        protected_attribute_names=["protected_bin"],
        scores_names=["score"],    # <-- necessário para o CalibratedEqOdds
        favorable_label=1,
        unfavorable_label=0
    )

    # Instancia e ajusta Calibrated Equalized Odds
    calib_eq_odds = CalibratedEqOddsPostprocessing(
        unprivileged_groups=[{'protected_bin': 0}],
        privileged_groups=[{'protected_bin': 1}],
        cost_constraint=params.get("cost_constraint")
    )

    calib_eq_odds = calib_eq_odds.fit(dataset_true, dataset_pred)

    # Transforma as previsões (preserva calibração)
    dataset_pred_transf = calib_eq_odds.predict(dataset_pred)

    # Extrai previsões corrigidas
    y_pred_transf = dataset_pred_transf.labels.ravel()

    return y_pred_transf, y_proba
