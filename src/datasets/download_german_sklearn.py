from sklearn.datasets import fetch_openml
import os

# baixa do OpenML (German Credit)
data = fetch_openml(name="credit-g", version=1, as_frame=True)

X, y = data.data, data.target
german_sklearn = X.copy()
german_sklearn["class"] = y  # alvo já é "class"

# cria coluna 'sex' baseada em 'personal_status'
german_sklearn["sex"] = german_sklearn["personal_status"].apply(
    lambda val: "female" if "female" in val else "male"
)

# salva dataset completo
german_sklearn.to_csv("german_sklearn.csv", index=False)

# calcula ponto de corte (80%)
n_total = len(german_sklearn)
n_train = int(n_total * 0.8)

# separa mantendo a ordem
train_df = german_sklearn.iloc[:n_train].copy()
test_df = german_sklearn.iloc[n_train:].copy()

# salva em CSV
train_df.to_csv("german_sklearn_train.csv", index=False)
test_df.to_csv("german_sklearn_test.csv", index=False)
