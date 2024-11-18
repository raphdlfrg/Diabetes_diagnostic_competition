import numpy as np
import pandas as pd
from time import time
import sklearn.preprocessing
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



x_train = pd.read_csv("data/train.csv")
x_test = pd.read_csv("data/test.csv")
y_train = pd.read_csv("data/labels.csv")

#x_train, x_test = x_train.drop(['Age_Group']), x_test.drop(['Age_Group'])


def feature_encoding(X):

    non_numerical_columns_names = X.select_dtypes(exclude=['number']).columns

    for column in non_numerical_columns_names:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    return X

def normalize_features(X_train, X_test):
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

x_train, x_test = feature_encoding(x_train), feature_encoding(x_test)

x_train_scaled, x_test_scaled = normalize_features(x_train, x_test)

model = RandomForestClassifier()

model.fit(x_train_scaled, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_f1 = f1_score(y_train, y_train_pred)

confusion_matrix_train = confusion_matrix(y_train, y_train_pred)

print(f"Training score for the model is {y_train_pred} and the confusion matrix is {confusion_matrix_train}")



