from sklearn.naive_bayes import BernoulliNB

def create_model(params: dict):
    return BernoulliNB(**params)