from sklearn.neighbors import KNeighborsClassifier

def create_model(params: dict):
    """Cria instância de K-Nearest Neighbors com parâmetros do YAML."""
    return KNeighborsClassifier(**params)