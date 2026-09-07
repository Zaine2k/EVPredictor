TARGET = "Will_Buy_EV"

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_PATH = "datasets/train.csv"
TEST_PATH = "datasets/test.csv"
SUBMISSION_PATH = "outputs/submission.csv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20