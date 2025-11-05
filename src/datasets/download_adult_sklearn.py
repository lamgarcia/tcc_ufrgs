from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd


data = fetch_openml(name="adult", version=2, as_frame=True)

X, y = data.data, data.target
adult_sklearn = X.copy()
adult_sklearn["income"] = y

adult_sklearn.to_csv("adult_sklearn.csv", index=False)

# Primeiro: 60% treino, 40% temporário
train_df, temp_df = train_test_split(
    adult_sklearn,
    test_size=0.4,
    stratify=adult_sklearn["income"],
    random_state=42
)

# Depois: 20% validação, 20% teste (50/50 do temporário)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["income"],
    random_state=42
)

train_df.to_csv("adult_sklearn_train.csv", index=False)
val_df.to_csv("adult_sklearn_val.csv", index=False)
test_df.to_csv("adult_sklearn_test.csv", index=False)

# --- Verifica proporções ---
def proporcao(df):
    return df["income"].value_counts(normalize=True).sort_index()

print("Proporção treino:\n", proporcao(train_df))
print("\nProporção validação:\n", proporcao(val_df))
print("\nProporção teste:\n", proporcao(test_df))

print("\nTamanhos:")
print(f"Treino: {len(train_df)} ({len(train_df)/len(adult_sklearn):.2%})")
print(f"Validação: {len(val_df)} ({len(val_df)/len(adult_sklearn):.2%})")
print(f"Teste: {len(test_df)} ({len(test_df)/len(adult_sklearn):.2%})")
print(f"Total: {len(adult_sklearn)} (100%)")
