import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o CSV (substitua 'seu_arquivo.csv' pelo nome do seu arquivo)
# Carregar o CSV
df = pd.read_csv('runs.csv')

# Criar coluna combinada: model + pre
df['model_pre'] = df['model'].astype(str) + '_' + df['pre'].astype(str)

# Criar gráfico
#plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=df,
    x='disparate_impact',
    y='accuracy',
    hue='model_pre',  # agora usa a combinação model_pre
    s=100,
    palette='colorblind'  # ou 'Set1', 'husl', etc. — escolha uma paleta com boas cores distintas
)

plt.title('Accuracy vs Disparate Impact ', fontsize=16)
plt.xlabel('Disparate Impact', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Fairness (DI = 1)')
plt.legend(title='Model_Pre', bbox_to_anchor=(1.05, 1), loc='upper left')  # leg
plt.tight_layout()

plt.show()


fairness_cols = [
    'disparate_impact',
    'statistical_parity_diff',
    'equalized_odds_diff',
    'average_odds_diff',
    'generalized_entropy_index'
]

corr_matrix = df[['accuracy'] + fairness_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlação entre Accuracy e Métricas de Fairness')
plt.show()


