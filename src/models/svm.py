from sklearn.svm import SVC

def create_model(params: dict):
    return SVC(**params)