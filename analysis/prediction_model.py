import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
import os

from db_connector import fetch_evaluation_data  # adjust this import as needed

MODEL_PATH = "analysis/satisfaction_model.joblib"

def prepare_features(df):
    # One-hot encode codeCentre and nom_centre
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = df[['codeCentre', 'nom_centre']]
    encoded = encoder.fit_transform(categorical_features).toarray()

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(['codeCentre', 'nom_centre']),
        index=df.index
    )

    # Keep only numeric target column
    df_final = pd.concat([encoded_df, df[['avis_num']]], axis=1)
    return df_final, encoder

def train_model():
    print("📦 Loading data from database...")
    df = fetch_evaluation_data()

    print("🧹 Cleaning & preprocessing...")
    df['avis_num'] = pd.to_numeric(df['avis'], errors='coerce')
    df.dropna(subset=['avis_num'], inplace=True)

    df_encoded, encoder = prepare_features(df)

    X = df_encoded.drop('avis_num', axis=1)
    y = df_encoded['avis_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🤖 Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"✅ R² Score: {r2_score(y_test, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ RMSE: {rmse:.4f}")


    # Save model
    os.makedirs("reports", exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
