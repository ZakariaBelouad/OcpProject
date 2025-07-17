import os
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from db_connector import fetch_evaluation_data

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_anomalies(df, method='zscore', threshold=2.0):
    # Preprocessing
    df['datetime'] = pd.to_datetime(df['dateEtHeure'])
    df['day'] = df['datetime'].dt.date
    df['avis_num'] = pd.to_numeric(df['avis'], errors='coerce')
    df = df.dropna(subset=['avis_num'])

    # Moyenne journalière par centre
    daily_avg = df.groupby(['codeCentre', 'day'])['avis_num'].mean().reset_index()
    daily_avg.rename(columns={'avis_num': 'avg_score'}, inplace=True)

    if method == 'zscore':
        daily_avg['zscore'] = daily_avg.groupby('codeCentre')['avg_score'].transform(zscore)
        daily_avg['anomaly'] = daily_avg['zscore'].abs() > threshold

    elif method == 'isolationforest':
        model = IsolationForest(contamination=0.1, random_state=42)
        results = []
        for center, group in daily_avg.groupby('codeCentre'):
            scores = group[['avg_score']].values
            model.fit(scores)
            preds = model.predict(scores)
            group['anomaly'] = preds == -1
            results.append(group)
        daily_avg = pd.concat(results)

    else:
        raise ValueError("Invalid method: use 'zscore' or 'isolationforest'.")

    return daily_avg[['codeCentre', 'day', 'avg_score', 'anomaly']]

def export_anomalies():
    df = fetch_evaluation_data()
    anomalies = detect_anomalies(df, method='zscore', threshold=1.5)
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)
    print("✅ anomalies.csv created in /reports")

if __name__ == "__main__":
    export_anomalies()
