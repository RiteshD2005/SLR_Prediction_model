import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "slr_stacking_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "slr_preprocessor.pkl"
)


# --------------------------------------------------
# LOAD MODEL AND PREPROCESSOR ONCE
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# PRESSURE FEATURE EXTRACTION
# SAME LOGIC AS TRAINING
# --------------------------------------------------

PRESSURE_COLUMN = "DEFL. TEST\nPRESSURES (PSI)"


def extract_pressure_features(dataframe):
    dataframe = dataframe.copy()

    def parse_pressures(value):

        if pd.isna(value):
            return pd.Series([
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ])

        try:

            values = [
                float(x.strip())
                for x in str(value).split(",")
                if x.strip() != ""
            ]

            if len(values) == 0:
                return pd.Series([
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                ])

            return pd.Series([
                min(values),
                max(values),
                np.mean(values),
                len(values)
            ])

        except Exception:

            return pd.Series([
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ])

    extracted = dataframe[PRESSURE_COLUMN].apply(
        parse_pressures
    )

    extracted.columns = [
        "DEFL_PRESSURE_MIN",
        "DEFL_PRESSURE_MAX",
        "DEFL_PRESSURE_MEAN",
        "DEFL_PRESSURE_COUNT"
    ]

    dataframe = dataframe.drop(
        columns=[PRESSURE_COLUMN]
    )

    dataframe = pd.concat(
        [
            dataframe.reset_index(drop=True),
            extracted.reset_index(drop=True)
        ],
        axis=1
    )

    return dataframe


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict(input_data: dict):

    dataframe = pd.DataFrame([input_data])

    # Same pressure preprocessing used during training
    dataframe = extract_pressure_features(dataframe)

    # Same categorical handling used during training
    categorical_columns = [
        "TYRE SIZE",
        "PLY MATERIAL",
        "PLY EPI",
        "RIM CODE",
        "TYRE"
    ]

    for col in categorical_columns:

        if col in dataframe.columns:

            dataframe[col] = (
                dataframe[col]
                .fillna("MISSING")
                .astype(str)
            )

    # Transform using the SAVED preprocessor
    processed_data = preprocessor.transform(
        dataframe
    )

    # Prediction
    prediction = model.predict(
        processed_data
    )

    return float(prediction[0])
