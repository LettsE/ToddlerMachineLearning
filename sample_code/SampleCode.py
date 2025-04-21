import os
import pandas as pd
import numpy as np
from pathlib import Path
import tsfresh
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
BASE_DIR = Path(__file__).resolve().parent


'''
This script contains sample code to demonstrate how the machine learning models developped by Letts et al. can be used.
When running this code, make sure this script is in the same folder as the input data (.csv) and model (.json) that you are using.
The input .csv file needs to be raw accelerometer data at 30Hz with the headings: Datetime,X,Y,X,vector_magnitude
'''


# Input file and output folder selections
input_file = pd.read_csv(BASE_DIR/"SampleData.csv") 
output_folder = BASE_DIR 


'''
CHOOSE WHICH MODEL YOU WANT TO APPLY
Uncomment the model and label_classes you want to use
'''
# # Model: SED/TPA
# model_path = BASE_DIR/"SED-0_TPA-1_5sIntPos.json"
# label_classes = ['SED', 'TPA']

# # Model: SED/LPA/MVPA
# model_path = BASE_DIR/"SED-0_LPA-1_MVPA-2_5sIntPos.json"
# label_classes = ['SED', 'LPA', 'MVPA']

# Model: NVM/SED/TPA
model_path = BASE_DIR/"NVM-0_SED-1_TPA-2_5sIntPos.json"
label_classes = ['NVM', 'SED', 'TPA']

# # Model: NVM/SED/LPA/MVPA
# model_path = BASE_DIR/"NVM-0_SED-1_LPA-2_MVPA-3_5sIntPos.json"
# label_classes = ['NVM', 'SED', 'LPA', 'MVPA']






# LOADING FUNCTIONS
# DICTIONARY OF FEATURES TO EXTRACT FROM RAW DATA
feature_dict = {
    "mean": None,
    "standard_deviation": None,
    "minimum": None,
    "maximum": None,
    "quantile": [
       {"q": 0.1},
       {"q": 0.25},
       {"q": 0.50},
       {"q": 0.75},
       {"q": 0.95},
    ],    
    "variation_coefficient": None,
    "sum_values": None,
    "median": None,
    "skewness": None,
    "kurtosis": None,
    "root_mean_square": None,
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"}
    ],
    "fourier_entropy": [{"bins": 10}],  
    "fft_coefficient": [
        {"coeff": 0, "attr": "real"},
        {"coeff": 1, "attr": "real"},
        {"coeff": 2, "attr": "real"},
        {"coeff": 3, "attr": "real"},
        {"coeff": 4, "attr": "real"},
        {"coeff": 0, "attr": "imag"},
        {"coeff": 1, "attr": "imag"},
        {"coeff": 2, "attr": "imag"},
        {"coeff": 3, "attr": "imag"},
        {"coeff": 4, "attr": "imag"},
        {"coeff": 0, "attr": "abs"},
        {"coeff": 1, "attr": "abs"},
        {"coeff": 2, "attr": "abs"},
        {"coeff": 3, "attr": "abs"},
        {"coeff": 4, "attr": "abs"},
        {"coeff": 0, "attr": "angle"},
        {"coeff": 1, "attr": "angle"},
        {"coeff": 2, "attr": "angle"},
        {"coeff": 3, "attr": "angle"},
        {"coeff": 4, "attr": "angle"}
    ]
}

# FUNCTION TO EXTRACT FEATURES WITH TIMES FROM RAW DATA
def extract_features_with_start_times(df, epoch=5, hz=30):
    group_size = epoch*hz  # Number of rows to process at once

    features_merged = pd.DataFrame()
    start_times = []

    start_index = 0

    for i in range(start_index, len(df), group_size):
        # Select group of rows
        group_df = df.iloc[i:i+group_size]
        start_times.append(df.iloc[i]['Datetime'])

        group_df = group_df.filter(regex=r'^(X|Y|Z|vector_magnitude)', axis=1)

        # Add a new column as the index
        group_df.insert(0, 'Index', range(1, len(group_df) + 1))
        group_df.insert(0, 'id', 1)

        # Extract features from the new DataFrame
        extracted_features = tsfresh.extract_features(group_df, column_id="id", column_sort='Index', n_jobs=1, default_fc_parameters=feature_dict, disable_progressbar=True)

        features_merged = pd.concat([features_merged, extracted_features], axis=0)

    return (features_merged, start_times)

# FUNCTION TO GET LABEL ENCODER NEEDED FOR XGBOOST CLASSIFIER
def get_label_encoder(classes):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

    return label_encoder

# FUNCTION TO LOAD XGBOOST MODEL
def load_xgboost_classifier(path):
    classifier = xgb.XGBClassifier()
    classifier.load_model(path)

    return classifier

# FUNCTION TO MAKE PREDICTIONS USING LOADED XGBOOST CLASSIFIER FROM EXTRACTED FEATURES
def predict(model, features, start_times, label_encoder):
    prediction = model.predict(features)

    features['Prediction'] = label_encoder.inverse_transform(prediction)
    features['Time'] = start_times

    return features













# CODE TO ANALYZE THE CSV
print("Loading model...")
model = load_xgboost_classifier(model_path)
label_encoder = get_label_encoder(label_classes)
print("Model loaded successfully.")

print("Processing file...")
data = input_file

print("Extracting features...")
features, start_times = extract_features_with_start_times(data, epoch=5, hz=30) #CHANGE THE HZ FOR DIFFERENT SAMPLING FREQUENCIES, ALL MODELS USE A 5s EPOCH
print("Features extracted successfully.")

print("Making predictions...")
predictions = predict(model, features, start_times, label_encoder)
predictions.to_csv(os.path.join(output_folder, 'predictions.csv'), index=False)
print("Processing complete!")