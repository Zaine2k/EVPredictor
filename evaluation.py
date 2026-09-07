from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    train_predictions = model.predict(X_train)
    valid_predictions = model.predict(X_valid)

    train_probabilities = model.predict_proba(X_train)[:, 1]
    valid_probabilities = model.predict_proba(X_valid)[:, 1]

    train_roc_auc = roc_auc_score(
        y_train,
        train_probabilities,
    )

    valid_roc_auc = roc_auc_score(
        y_valid,
        valid_probabilities,
    )

    print(f"Training ROC-AUC:   {train_roc_auc:.4f}")
    print(f"Validation ROC-AUC: {valid_roc_auc:.4f}")

    print("\nValidation Classification Report:")
    print(
        classification_report(
            y_valid,
            valid_predictions,
            target_names=["No", "Yes"],
        )
    )

    print("Validation Confusion Matrix:")
    print(
        confusion_matrix(
            y_valid,
            valid_predictions,
        )
    )