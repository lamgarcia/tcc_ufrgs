from sklearn.datasets import fetch_openml

# baixa do OpenML
data = fetch_openml(name="adult", version=2, as_frame=True)

X, y = data.data, data.target
adult_sklearn = X.copy()
adult_sklearn["income"] = y

# salva em CSV
adult_sklearn.to_csv("adult_sklearn/adult_sklearn.csv", index=False)
print("adult_sklearn.csv")