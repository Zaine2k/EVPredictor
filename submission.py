import pandas as pd

from config import SUBMISSION_PATH, TARGET


def create_submission(model, X_test, test_ids):
    # Probability that Will_Buy_EV is Yes (class 1)
    probabilities = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "id": test_ids.reset_index(drop=True),
        TARGET: probabilities,
    })

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    return submission