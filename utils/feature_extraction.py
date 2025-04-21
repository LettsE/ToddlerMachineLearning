import pandas as pd
import tsfresh

# #features to extract
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

