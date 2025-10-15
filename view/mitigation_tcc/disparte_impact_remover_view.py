import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover

# -----------------------
# 1. Carregar dataset
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "adult_sklearn_train.csv")
df = pd.read_csv(csv_path)

protected = "sex"
feature = "hours-per-week"

df = df[[protected, feature]].dropna()

# -----------------------
# 2. Função auxiliar para plotar gráficos
# -----------------------
def plot_distributions(male, female, title_suffix="Original"):
    plt.figure(figsize=(14, 4))

    # Hist side by side
    plt.subplot(1, 3, 1)
    plt.hist(male, bins=30, alpha=0.7, color="blue", density=True)
    plt.title(f"Distribuição Male ({title_suffix})")
    plt.xlabel(feature)
    plt.ylabel("Density")

    plt.subplot(1, 3, 2)
    plt.hist(female, bins=30, alpha=0.7, color="orange", density=True)
    plt.title(f"Distribuição Female ({title_suffix})")
    plt.xlabel(feature)

    # CDF
    plt.subplot(1, 3, 3)
    for data, label, color in [(male, "Male", "blue"), (female, "Female", "orange")]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=label, color=color)
    plt.xlabel(feature)
    plt.ylabel("CDF")
    plt.title(f"CDF ({title_suffix})")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Função Quantílica
    plt.figure(figsize=(6, 4))
    quantiles = np.linspace(0, 1, 100)
    male_q = np.quantile(male, quantiles)
    female_q = np.quantile(female, quantiles)
    plt.plot(quantiles, male_q, label="Male", color="blue")
    plt.plot(quantiles, female_q, label="Female", color="orange")
    plt.xlabel("Percentil")
    plt.ylabel(feature)
    plt.title(f"Função Quantílica ({title_suffix})")
    plt.legend()
    plt.show()

# -----------------------
# 3. Gráficos antes da reparação
# -----------------------
# Mapear masculino/feminino para números
df[protected] = df[protected].map({"Male": 1, "Female": 0})

# Agora você ainda pode separar para o plot
male = df[df[protected] == 1][feature]
female = df[df[protected] == 0][feature]
plot_distributions(male, female, "Original")

# -----------------------
# 4. Aplicar Disparate Impact Remover
# -----------------------
    # Cria BinaryLabelDataset
dataset = BinaryLabelDataset(
        df=df,
        label_names=feature,
        protected_attribute_names=[protected],
        favorable_label=1,
        unfavorable_label=0
 )

# Reparação (repair_level=1.0 aplica máxima normalização)
dir = DisparateImpactRemover(repair_level=1.0)
dataset_repaired = dir.fit_transform(dataset)

# Reconstruir DataFrame
df_repaired = dataset_repaired.convert_to_dataframe()[0]
male_r = df_repaired[df_repaired[protected] == "Male"][feature]
female_r = df_repaired[df_repaired[protected] == "Female"][feature]

# -----------------------
# 5. Gráficos após reparação
# -----------------------
plot_distributions(male_r, female_r, "Reparado (DIR)")
