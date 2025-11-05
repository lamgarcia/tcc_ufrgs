from sklearn.datasets import fetch_openml
import os

# baixa do OpenML
data = fetch_openml(name="adult", version=2, as_frame=True)

X, y = data.data, data.target
adult_sklearn = X.copy()
adult_sklearn["income"] = y

# cria diretório se não existir
#os.makedirs("adult_sklearn", exist_ok=True)
adult_sklearn.to_csv("adult_sklearn.csv", index=False)

# calcula ponto de corte (80%)
n_total = len(adult_sklearn)
n_train = int(n_total * 0.8)

# separa mantendo a ordem
train_df = adult_sklearn.iloc[:n_train].copy()
test_df = adult_sklearn.iloc[n_train:].copy()

# salva em CSV
train_df.to_csv("adult_sklearn_train.csv", index=False)
test_df.to_csv("adult_sklearn_test.csv", index=False)
