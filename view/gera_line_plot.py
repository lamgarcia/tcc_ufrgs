import pandas as pd
import matplotlib.pyplot as plt

# === Lê o CSV ===
df = pd.read_csv("runs.csv")

# Métricas de interesse
metrics = [
    "accuracy",
    "statistical_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "average_predictive_value_difference",
    "disparate_impact"
]

# Lista de modelos
models = df['model'].unique()

# Para cada modelo
for model in models:
    df_model = df[df['model'] == model].copy()
    
    # Cria coluna pipeline
    df_model['pipeline'] = df_model['pre'] + "_" + df_model['in'] + "_" + df_model['post']
    
    # Conta quantas técnicas foram aplicadas (0 = baseline)
    df_model['num_techniques'] = df_model[['pre','in','post']].apply(lambda x: sum([1 for v in x if v != 'none']), axis=1)
    
    # Ordena por número de técnicas aplicadas, depois alfabética para consistência
    df_model = df_model.sort_values(by=['num_techniques','pipeline'])
    
    # Cria gráfico
    plt.figure(figsize=(12,6))
    
    for metric in metrics:
        plt.plot(df_model['pipeline'], df_model[metric], marker='o', label=metric)
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Pipeline (pre_in_post)')
    plt.ylabel('Valor da Métrica')
    plt.title(f"Evolução das Métricas - Modelo: {model}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
