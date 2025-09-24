import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "runs_adult.csv")
df = pd.read_csv(csv_path)

# === Extrair colunas necessárias ===
x = df["equalized_odds_difference"]
y = df["average_predictive_value_difference"]
z = df["statistical_parity_difference"]

# === Criar gráfico 3D ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter 3D
sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=60, alpha=0.8, edgecolor="k")

# Rótulos dos eixos
ax.set_xlabel("equalized_odds_difference")
ax.set_ylabel("average_predictive_value_difference")
ax.set_zlabel("Statistical Parity Difference")

# Barra de cores
plt.colorbar(sc, ax=ax, label="Statistical Parity Difference")

plt.title("")
plt.show()
