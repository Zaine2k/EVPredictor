import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_SIZE,
)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def encode_target(target):
    encoded_target = target.map({
        "No": 0,
        "Yes": 1,
    })

    if encoded_target.isna().any():
        unexpected_values = target[encoded_target.isna()].unique()

        raise ValueError(
            f"Unexpected or missing target values: {unexpected_values}"
        )

    return encoded_target


def prepare_training_data(train_df):
    X = train_df[FEATURES]
    y = encode_target(train_df[TARGET])

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def prepare_test_data(test_df):
    return test_df[FEATURES]