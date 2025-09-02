# src/features/preprocess.py
import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Columns expected (based on your sample)
NUM_FEATURES = [
    'Age','Annual Income','Number of Dependents','Health Score',
    'Previous Claims','Vehicle Age','Credit Score','Insurance Duration'
]
CAT_FEATURES = [
    'Gender','Marital Status','Education Level','Occupation',
    'Location','Policy Type','Customer Feedback','Smoking Status',
    'Exercise Frequency','Property Type'
]

class PolicyDateCleaner(BaseEstimator, TransformerMixin):
    """Try to parse 'Policy Start Date' and extract year; invalids become -1."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'Policy Start Date' in X.columns:
            X['Policy Start Date'] = pd.to_datetime(X['Policy Start Date'], dayfirst=True, errors='coerce')
            X['policy_start_year'] = X['Policy Start Date'].dt.year.fillna(-1).astype(int)
            X = X.drop(columns=['Policy Start Date'])
        return X

def _onehot_encoder_compat(**kwargs):
    """
    Create OneHotEncoder with compatibility across sklearn versions:
    older: OneHotEncoder(sparse=False), newer: sparse_output=False
    """
    v = sklearn.__version__.split('.')
    major, minor = int(v[0]), int(v[1]) if len(v) > 1 else 0
    if (major, minor) >= (1, 2):  # sklearn >= 1.2 uses sparse_output
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False, **kwargs)
    else:
        return OneHotEncoder(handle_unknown='ignore', sparse=False, **kwargs)

def build_preprocessor(num_features=None, cat_features=None):
    if num_features is None:
        num_features = NUM_FEATURES
    if cat_features is None:
        cat_features = CAT_FEATURES

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', _onehot_encoder_compat())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='drop')

    full_pipeline = Pipeline([
        ('date_cleaner', PolicyDateCleaner()),
        ('preprocessor', preprocessor)
    ])
    return full_pipeline
