from sklearn.linear_model import LogisticRegression

def create_model(params: dict):
    """Cria instância de Regressão Logística com parâmetros do YAML."""
    return LogisticRegression(**params)