import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

def get_label_encoder(classes):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

    return label_encoder


def load_xgboost_classifier(path):
    classifier = xgb.XGBClassifier()
    classifier.load_model(path)

    return classifier


def predict(model, features, start_times, label_encoder):
    prediction = model.predict(features)

    features['Prediction'] = label_encoder.inverse_transform(prediction)
    features['Time'] = start_times

    return features