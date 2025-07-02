# src/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_raw_data(path='../data/raw/healthcare-dataset-stroke-data.csv'):
    df = pd.read_csv(path)
    df = df.drop(columns=['id'])  # Drop ID
    return df

def preprocess_and_split(df):
    target = 'stroke'
    X = df.drop(columns=[target])
    y = df[target]

    # Define columns
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Fit/transform the data
    X_processed = preprocessor.fit_transform(X)

    # Save the transformer
    joblib.dump(preprocessor, '../models/preprocessor.pkl')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_raw_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    print(X_train.shape)
    print(X_test.shape)