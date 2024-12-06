# Kaggle Competion - Diabetes Detection

This project implements a logistic regression model using `scikit-learn` with an optimized decision threshold for binary classification. The model's hyperparameters are tuned using `GridSearchCV` with `StratifiedKFold` cross-validation. The decision threshold is optimized for maximum F1-score, making it suitable for imbalanced datasets or tasks where precision-recall trade-offs are critical.

---

## Features

- **Logistic Regression**:
  - ElasticNet regularization.
  - Class-weight balancing for handling class imbalance.
  - Hyperparameter optimization using `GridSearchCV`.

- **Threshold Optimization**:
  - Custom threshold applied to the probabilities (`predict_proba`).
  - F1-score maximized for thresholds ranging from 0.1 to 0.9.

- **Evaluation**:
  - Training set evaluation with the best hyperparameters for each threshold.
  - Generates detailed metrics (F1-score, precision, recall, classification report).

---

## Requirements

Ensure you have the following dependencies installed:

```bash
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
```

---

## Files

- **`log_regV2_with_threshold.py`**: The main Python script that implements the logistic regression model, hyperparameter tuning, and threshold optimization.

---

## Usage

1. **Data Preparation**:
   - Ensure the datasets (`train.csv`, `labels.csv`, and `test.csv`) are placed in the `data/` directory.
   - The script expects:
     - `train.csv` and `labels.csv` for training features and labels.
     - `test.csv` for test features.

2. **Run the Script**:
   Execute the Python script to train the model, optimize hyperparameters, and predict test labels.

   ```bash
   python log_regV2_with_threshold.py
   ```

3. **Outputs**:
     - **`test_predictionsLogReg_CustomThreshold.csv`**:
     - Predicted labels for the test set with the optimized threshold.

---

## Code Overview

1. **Feature Encoding**:
   - Non-numerical features are encoded using `LabelEncoder`.

2. **Data Normalization**:
   - All features are standardized using `StandardScaler`.

3. **GridSearchCV with Threshold**:
   - Hyperparameters are tuned for multiple thresholds.
   - The F1-score is computed for each threshold during cross-validation.

4. **Training and Evaluation**:
   - The model is evaluated on the training set with the best hyperparameters for each threshold.
   - Threshold optimization maximizes the F1-score for better precision-recall trade-offs.

5. **Test Predictions**:
   - Test set predictions are generated using the best threshold.

---

## Customization

- **Adjust Thresholds**:
  Modify the range of thresholds to evaluate by changing the `thresholds` array:

  ```python
  thresholds = np.linspace(0.595, 0.61, 10)
  ```

- **Modify Hyperparameters**:
  Update the `param_grid` dictionary to explore other hyperparameter values.

- **Evaluation Metric**:
  Change the metric used in `GridSearchCV` or custom scoring function as needed.
