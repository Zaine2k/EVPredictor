from data import (
    load_data,
    prepare_test_data,
    prepare_training_data,
    split_data,
)
from evaluation import evaluate_model
from pipeline import create_model_pipeline
from submission import create_submission


def main():
    # Load data
    train_df, test_df = load_data()

    # Prepare training data
    X, y = prepare_training_data(train_df)

    # Create validation split
    X_train, X_valid, y_train, y_valid = split_data(X, y)

    # Create and train the model
    model = create_model_pipeline()
    model.fit(X_train, y_train)

    # Evaluate validation performance
    evaluate_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    # Retrain using all available training data
    model.fit(X, y)

    # Generate test predictions
    X_test = prepare_test_data(test_df)

    submission = create_submission(
        model,
        X_test,
        test_df["id"],
    )

    print("\nSubmission preview:")
    print(submission.head())

    print("\nSaved predictions to submission.csv")


if __name__ == "__main__":
    main()