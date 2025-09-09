import pandas as pd

# nomes de colunas segundo a documentação oficial
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# URLs da UCI
train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
test_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# carrega datasets
train = pd.read_csv(train_url, names=columns, sep=",", skipinitialspace=True)
test  = pd.read_csv(test_url, names=columns, sep=",", skiprows=1, skipinitialspace=True)

# remove ponto final nos rótulos do conjunto de teste
test["income"] = test["income"].str.replace(".", "", regex=False)

# junta tudo
adult_uci = pd.concat([test, train], ignore_index=True)

# salva em CSV
adult_uci.to_csv("adult_uci/adult_uci.csv", index=False)
print("adult_uci.csv")
