from sklearn.ensemble import RandomForestClassifier

def create_model(params: dict):
    """Cria instância de Random Forest com parâmetros do YAML."""
    return RandomForestClassifier(**params)