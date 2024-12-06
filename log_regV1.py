import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

SEED = 1
np.random.seed(SEED)

# Charger les données
train_data = pd.read_csv('data/train.csv')
labels_data = pd.read_csv('data/labels.csv')
test_data = pd.read_csv('data/test.csv')

# Suppression certaines features/colonnes
X_train = train_data.drop(columns=['Unnamed: 0', 'Age_Group'])
y_train = labels_data
X_test = test_data.drop(columns=['Unnamed: 0', 'Age_Group'])


def feature_encoding(X):
    """
    One-hot encode the 'features'.
    Input: X: features (pd.DataFrame) with shape = (45211, 16)
    Output: X: features_encoded (pd.DataFrame) with shape = (45211, 16)
    """
    non_numerical_columns_names = X.select_dtypes(exclude=['number']).columns
    for column in non_numerical_columns_names:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    return X


# Encodage features non-numérique
X_train = feature_encoding(X_train)
X_test = feature_encoding(X_test)

# Normalisation des features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=SEED)
X_train_S, y_train_S = smote.fit_resample(X_train, y_train['Diabetes_binary'])


param_grid = {
    'penalty': ['elasticnet'],
    'C': [0.001, 0.005, 0.01],
    'solver': ['saga'],  # Supports 'l1' and 'elasticnet'
    #'max_iter': [100, 500, 1000],
    'class_weight': ['balanced'],
    'l1_ratio': [0.5],  # For 'elasticnet'
}


def perform_grid_search(model, X_train, Y_train, params):
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(model, param_grid=params, scoring='f1', cv=strat_kfold)
    grid_search.fit(X_train, Y_train)

    best_param = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters are:", best_param)
    print("Best score is:", best_score)
    return grid_search, best_param, best_score

# cls = LogisticRegression(random_state=SEED)
# grid_search_rf, best_param_rf, best_score_rf = perform_grid_search(cls, X_train, y_train['Diabetes_binary'], params=param_grid)

#param_grid = {'C': [0.005], 'class_weight': ['balanced'], 'l1_ratio': [0.5], 'penalty': ['elasticnet'], 'solver': ['saga']}
#param_grid = {'C': [1]}
cls = LogisticRegression(random_state=SEED)
grid_search_rf, best_param_rf, best_score_rf = perform_grid_search(cls, X_train, y_train['Diabetes_binary'], params=param_grid)


cls_final = LogisticRegression(
    C=best_param_rf['C'],
    class_weight=best_param_rf['class_weight'],
    l1_ratio=best_param_rf['l1_ratio'],
    #max_iter=best_param_rf['max_iter'],
    penalty=best_param_rf['penalty'],
    solver=best_param_rf['solver'],
    random_state=SEED
)

cls_final = LogisticRegression(random_state=SEED)
cls_final.fit(X_train, y_train['Diabetes_binary'])

y_test_pred = cls_final.predict(X_test)

print(y_test_pred.shape)

y_test_pred = pd.DataFrame(y_test_pred, columns=['Diabetes_binary'], index=test_data['Unnamed: 0'])

y_test_pred.index.name = 'index'

y_test_pred.to_csv("test_predictionsLogReg2.csv", index=True)



"""

# Initialize logistic regression and grid search
#log_reg = LogisticRegression(random_state=42)
#grid_search = GridSearchCV(log_reg, param_grid, scoring='f1', cv=5, verbose=1, n_jobs=-1)

# Fit grid search
#grid_search.fit(X_train, y_train['Diabetes_binary'])


# Step 4: Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
model = LogisticRegression(random_state=42, max_iter=1000)

f1_scores = []
fold_sizes = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold + 1}")

    # Split data into training and validation for this fold
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    fold_sizes.append(X_train_fold.shape)

    # Print the distribution of classes in each fold
    print("Class Distribution in Fold:")
    print(y_train_fold.value_counts(normalize=True))

    # Train and validate the model
    model.fit(X_train_fold, y_train_fold)
    y_val_pred = model.predict(X_val_fold)
    f1 = f1_score(y_val_fold, y_val_pred)
    f1_scores.append(f1)

    print(f"F1-Score for Fold {fold + 1}: {f1:.2f}")
    print(classification_report(y_val_fold, y_val_pred))

# Average F1-Score across all folds
mean_f1 = np.mean(f1_scores)
print(f"\nAverage F1-Score across all folds: {mean_f1:.2f}")

# Step 5: Train on Full Training Set and Predict on Test Data
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

print(f"\nPredicted Labels for Test Data:\n{test_predictions}")
"""