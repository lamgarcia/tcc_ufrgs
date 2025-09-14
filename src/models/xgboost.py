from xgboost import XGBClassifier

def create_model(params: dict):
    return XGBClassifier(**params)