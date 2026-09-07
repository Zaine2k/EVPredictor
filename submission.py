import pandas as pd

from config import SUBMISSION_PATH, TARGET


def create_submission(model, X_test, test_ids):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    labels = pd.Series(predictions).map({
        0: "No",
        1: "Yes",
    })

    submission = pd.DataFrame({
        "id": test_ids.reset_index(drop=True),
        TARGET: labels,
        "Purchase_Probability": probabilities,
    })

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    return submission