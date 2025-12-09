import pandas as pd

# === 1. Ler CSV ===
df = pd.read_csv("runs_adult_30x_2.csv")

# === 2. Chave única de agrupamento ===
group_keys = ["model", "pre", "in", "post"]

# === 3. Remover colunas "id" e "data"
df = df.drop(columns=["id", "data"], errors="ignore")

# === 4. Identificar colunas que serão agregadas (todas exceto keys)
agg_cols = [c for c in df.columns if c not in group_keys]

# === 5. Criar dicionário de agregação: mean + std
agg_dict = {col: ["mean", "std"] for col in agg_cols}

# === 6. Agrupar e calcular ===
grouped = df.groupby(group_keys).agg(agg_dict)

# === 7. Ajustar nomes das colunas ===
new_cols = []
for col, stat in grouped.columns:
    if stat == "mean":
        new_cols.append(col)          # média mantém o nome
    else:
        new_cols.append(f"{col}_std") # desvio vira col_std
grouped.columns = new_cols

# === 8. Resetar índice para trazer model/pre/in/post de volta ===
grouped = grouped.reset_index()

# === 9. Reordenar: cada métrica seguida de sua _std ===
metric_pairs = []
for col in agg_cols:
    metric_pairs.append(col)
    metric_pairs.append(f"{col}_std")

final_cols = group_keys + metric_pairs
grouped = grouped[final_cols]

# === 10. Salvar CSV final ===
grouped.to_csv("runs_adult_30x_2_mean_std.csv", index=False)

print("Arquivo gerado com sucesso!")
