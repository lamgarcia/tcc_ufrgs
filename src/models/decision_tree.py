from sklearn.tree import DecisionTreeClassifier

def create_model(params: dict):
    return DecisionTreeClassifier(**params)