import os
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
from scipy.stats import zscore
from db_connector import fetch_evaluation_data

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# === LOAD DATA ===
def load_data():
    df = fetch_evaluation_data()
    df['datetime'] = pd.to_datetime(df['dateEtHeure'])
    df['day'] = df['datetime'].dt.date
    return df

# === EDA PLOTS ===
def plot_daily_average(df):
    daily_avg = df.groupby('day')['avis'].mean()
    plt.figure(figsize=(6, 4))
    daily_avg.plot(marker='o')
    plt.title("Daily Average Satisfaction")
    plt.xlabel("Day")
    plt.ylabel("Average Score")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "daily_avg.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_center_average(df):
    center_avg = df.groupby('codeCentre')['avis'].mean()
    plt.figure(figsize=(6, 4))
    center_avg.plot(kind='bar', color='skyblue')
    plt.title("Center-wise Average Satisfaction")
    plt.xlabel("Center")
    plt.ylabel("Average Score")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "center_avg.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_distribution(df):
    plt.figure(figsize=(6, 4))
    df['avis'].plot(kind='hist', bins=4, rwidth=0.8, color='salmon')
    plt.title("Satisfaction Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "distribution.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_heatmap(df):
    heatmap_data = df.groupby(['day', 'codeCentre'])['avis'].mean().unstack()
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Avg Satisfaction')
    plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns, rotation=45)
    plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index)
    plt.title("Heatmap of Satisfaction by Day and Center")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "heatmap.png")
    plt.savefig(path)
    plt.close()
    return path

def plot_pie(df):
    counts = df['avis'].value_counts().sort_index()
    label_map = {4: "Très Satisfait", 3: "Satisfait", 2: "Peu Satisfait", 1: "Pas du tout Satisfait"}
    labels = [label_map.get(val, str(val)) for val in counts.index]
    plt.figure(figsize=(6, 4))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Répartition des avis")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "pie_chart.png")
    plt.savefig(path)
    plt.close()
    return path

# === ANOMALY DETECTION ===
def detect_anomalies(df, threshold=2):
    df['avis_num'] = pd.to_numeric(df['avis'], errors='coerce')
    df = df.dropna(subset=['avis_num'])
    daily_avg = df.groupby(['codeCentre', 'day'])['avis_num'].mean().reset_index()
    daily_avg.rename(columns={'avis_num': 'avg_score'}, inplace=True)
    daily_avg['zscore'] = daily_avg.groupby('codeCentre')['avg_score'].transform(zscore)
    daily_avg['anomaly'] = daily_avg['zscore'].abs() > threshold
    anomalies = daily_avg[daily_avg['anomaly']]
    return anomalies[['codeCentre', 'day', 'avg_score']]

# === PDF REPORT ===
def create_pdf(plots, anomalies_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Weekly Satisfaction Report", ln=True, align='C')
    pdf.ln(10)

    # Add charts
    for title, img_path in plots:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.image(img_path, w=180)
        pdf.ln(5)

    # Add anomalies if any
    if not anomalies_df.empty:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detected Anomalies", ln=True, align='L')
        pdf.set_font("Arial", "", 12)
        pdf.ln(5)
        for _, row in anomalies_df.iterrows():
            pdf.cell(0, 10, f"Center: {row['codeCentre']} | Date: {row['day']} | Score: {row['avg_score']:.2f}", ln=True)

    pdf.output(os.path.join(REPORT_DIR, "weekly_report.pdf"))

# === MAIN ===
def run_report():
    df = load_data()
    plots = [
        ("1. Daily Average Satisfaction", plot_daily_average(df)),
        ("2. Center-wise Average Satisfaction", plot_center_average(df)),
        ("3. Satisfaction Distribution", plot_distribution(df)),
        ("4. Heatmap of Satisfaction by Day and Center", plot_heatmap(df)),
        ("5. Pie Chart of Satisfaction Levels", plot_pie(df))
    ]
    anomalies_df = detect_anomalies(df, threshold=1.5)
    create_pdf(plots, anomalies_df)
    print("✅ PDF weekly report created.")

if __name__ == "__main__":
    run_report()
