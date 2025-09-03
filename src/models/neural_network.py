from sklearn.neural_network import MLPClassifier

def create_model(params: dict):
    """Cria instância de rede neural (MLP) com parâmetros do YAML."""
    return MLPClassifier(**params)