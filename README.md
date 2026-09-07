# EV Purchase Prediction with XGBoost

A machine learning project for predicting whether a customer will purchase an electric vehicle. The solution uses an XGBoost binary classifier with a reusable scikit-learn preprocessing pipeline for numeric and categorical data.

## Competition objective

The goal is to predict `Will_Buy_EV` for each customer in the test dataset.

| Target value | Meaning |
| --- | --- |
| `No` | The customer is not predicted to buy an EV |
| `Yes` | The customer is predicted to buy an EV |

During training, the target labels are encoded as `No = 0` and `Yes = 1`. Predictions are converted back to the original labels for submission.

## Dataset

The model uses the following features.

### Numeric features

- `Age`
- `Annual_Income_USD`
- `Daily_Commute_km`
- `Number_of_Cars_Owned`
- `Charging_Stations_Near_Home`
- `Charging_Stations_Near_Work`
- `Environmental_Concern_Level`

### Categorical features

- `Gender`
- `City_Type`
- `Current_Car_Type`
- `Home_Charging_Possible`
- `Subsidy_Available`
- `Range_Anxiety_Level`

The `id` column is retained for the submission file but is not used as a model input.

## Approach

The workflow consists of:

1. Loading the training and test datasets.
2. Encoding the `Yes` and `No` target labels.
3. Creating a stratified 80/20 training-validation split.
4. Replacing missing numeric values with the median.
5. Replacing missing categorical values with the most frequent value.
6. One-hot encoding categorical features while safely handling unseen categories.
7. Training an XGBoost binary classifier.
8. Evaluating the model using ROC-AUC, precision, recall, F1-score, and a confusion matrix.
9. Retraining the pipeline on all labeled data.
10. Generating predictions for the competition test set.

All preprocessing is contained in the same scikit-learn pipeline as the model. This ensures that training and test data receive identical transformations.

## Validation results

The current model produced the following results on the held-out validation set:

| Metric | Score |
| --- | ---: |
| Training ROC-AUC | 0.9419 |
| Validation ROC-AUC | 0.9410 |

The small difference between training and validation ROC-AUC suggests that the model is generalizing well rather than substantially overfitting.

### Confusion matrix

|  | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| Actual No | 104,547 | 5,830 |
| Actual Yes | 7,847 | 15,509 |

The model identifies most negative cases correctly, but misses some positive cases. Further work could focus on class weighting, decision-threshold tuning, and cross-validation to improve recall for likely EV buyers.

## Project structure

```text
project/
├── datasets/
│   ├── train.csv
│   └── test.csv
├── config.py
├── data.py
├── pipeline.py
├── evaluation.py
├── submission.py
├── main.py
├── requirements.txt
└── README.md
```

| File | Responsibility |
| --- | --- |
| `config.py` | Stores paths, feature lists, the target name, and shared settings |
| `data.py` | Loads data, encodes the target, selects features, and creates the validation split |
| `pipeline.py` | Builds the preprocessing steps and XGBoost pipeline |
| `evaluation.py` | Calculates and displays validation metrics |
| `submission.py` | Converts predictions to labels and saves the submission file |
| `main.py` | Runs the complete training, evaluation, and prediction workflow |

## Running the model

Place `train.csv` and `test.csv` inside the `datasets` directory, then run:

```bash
python main.py
```

The script will:

- train the model on the training split;
- print validation results;
- retrain the model using the full training dataset; and
- save predictions to the location defined by `SUBMISSION_PATH` in `config.py`.

The default path is:

```python
SUBMISSION_PATH = "submission.csv"
```

To save submissions in a separate directory, change it to:

```python
SUBMISSION_PATH = "outputs/submission.csv"
```

## Submission format

The generated file currently contains:

```csv
id,Will_Buy_EV,Purchase_Probability
668665,No,0.011889
668666,No,0.018883
```

Before uploading, compare the columns with Kaggle's provided `sample_submission.csv`. If Kaggle expects only `id` and `Will_Buy_EV`, remove `Purchase_Probability` from the submitted file.

## Model configuration

The current XGBoost classifier uses:

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
```

