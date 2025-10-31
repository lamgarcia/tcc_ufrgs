from sklearn.ensemble import RandomForestClassifier

def create_model(params: dict):
    return RandomForestClassifier(**params)