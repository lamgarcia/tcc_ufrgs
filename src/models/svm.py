from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def create_model(params: dict):
    return LinearSVC(**params)