import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, make_scorer

SEED = 1
np.random.seed(SEED)

# Load dataset
train_data = pd.read_csv('data/train.csv')
labels_data = pd.read_csv('data/labels.csv')
test_data = pd.read_csv('data/test.csv')

# Prepare features and
X_train = train_data.drop(columns=['Unnamed: 0', 'Age_Group'])
y_train = labels_data['Diabetes_binary']  # Extract target as Series
X_test = test_data.drop(columns=['Unnamed: 0', 'Age_Group'])


# Encode non-numerical features
def feature_encoding(X):
    non_numerical_columns_names = X.select_dtypes(exclude=['number']).columns
    for column in non_numerical_columns_names:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    return X

X_train = feature_encoding(X_train)
X_test = feature_encoding(X_test)


# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter grid
param_grid = {
    'C': [0.005, 0.0095, 0.01, 0.02],
    'penalty': ['elasticnet'],
    'solver': ['saga'],
    'l1_ratio': [0.45, 0.5, 0.55],
    'class_weight': ['balanced'],
}

#param_grid = {'C': [0.005], 'class_weight': ['balanced'], 'l1_ratio': [0.5], 'penalty': ['elasticnet'], 'solver': ['saga']}


# Custom threshold
def custom_f1_threshold(estimator, X, y, threshold=0.5):
    """Custom scorer for F1-score with adjustable threshold."""
    probs = estimator.predict_proba(X)[:, 1]  # Probabilities for the positive class
    preds = (probs >= threshold).astype(int)  # Apply the threshold
    return f1_score(y, preds)


# Perform GridSearchCV with StratifiedKFold
def find_best_params_for_thresholds(model, X_train, y_train, params, thresholds):
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for threshold in thresholds:
        print(f"\nEvaluating threshold: {threshold:.2f}")
        grid_search = GridSearchCV( model, param_grid=params, scoring=lambda est, X, y: custom_f1_threshold(est, X, y, threshold), cv=strat_kfold)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters for threshold {threshold:.2f}: {best_params}")
        print(f"Best F1-Score for threshold {threshold:.2f}: {best_score:.2f}")

        results.append({'threshold': threshold, 'best_params': best_params, 'best_score': best_score})

    return results

# Train with GridSearchCV using the different threshold
cls = LogisticRegression(random_state=SEED)
threshold = np.linspace(0.595, 0.61, 10)
#threshold = np.linspace(0.62, 0.62, 1)# Set your desired threshold here
results = find_best_params_for_thresholds(cls, X_train, y_train, param_grid, threshold)

best_f1_score = 0
best_index = 0
for i in range(len(results)):
    best = results[i]['best_score']
    if best > best_f1_score:
        best_f1_score = best
        best_index = i

best_results = results[best_index]
print("\n\n ++++++++++++++++++++++++++++++++++++++++ \n Best Score is", best_f1_score)
print("Threshold is", best_results['threshold'])
print(best_results)

# Training with the best values
threshold = best_results['threshold']
best_params = best_results['best_params']

# Train the model with the best hyperparameters
model = LogisticRegression(**best_params, random_state=SEED)
model.fit(X_train, y_train)

# Predict probabilities and apply the threshold
y_train_probs = model.predict_proba(X_train)[:, 1]
y_train_pred = (y_train_probs >= threshold).astype(int)

# Evaluate on training set
f1 = f1_score(y_train, y_train_pred)
report = classification_report(y_train, y_train_pred, output_dict=True)

print(f"F1-Score on training set for threshold {threshold:.2f}: {f1:.2f}")
print("Classification Report:")
print(classification_report(y_train, y_train_pred))

# Predict probabilities on the test set
y_test_probs = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_probs >= threshold).astype(int)

# Save predictions
y_test_pred = pd.DataFrame(y_test_pred, columns=['Diabetes_binary'], index=test_data['Unnamed: 0'])
y_test_pred.index.name = 'index'
y_test_pred.to_csv("test_predictionsLogReg_CustomThresholdTEMPO.csv", index=True)

print(f"Predictions saved to 'test_predictionsLogReg_CustomThreshold.csv'")
