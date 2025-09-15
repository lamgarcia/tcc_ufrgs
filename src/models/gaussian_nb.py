from sklearn.naive_bayes import GaussianNB

def create_model(params: dict):
    return GaussianNB(**params)