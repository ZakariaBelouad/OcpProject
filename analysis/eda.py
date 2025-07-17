# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_connector import fetch_evaluation_data

# Map numerical scores to labels
reverse_map = {
    4: "Très satisfait",
    3: "Satisfait",
    2: "Peu satisfait",
    1: "Pas du tout satisfait"
}

def preprocess(df):
    print("\n🧠 Initial data sample:")
    print(df.head())
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['dateEtHeure'])
    df['day'] = df['datetime'].dt.date
    
    # Ensure avis is numeric and map to labels
    df['avis_num'] = pd.to_numeric(df['avis'], errors='coerce')
    df = df.dropna(subset=['avis_num'])
    df['avis_label'] = df['avis_num'].map(reverse_map)
    
    print("\n✅ After mapping and cleaning:")
    print(df[['avis', 'avis_num', 'avis_label']].drop_duplicates())
    
    return df

def plot_daily_average(df):
    daily_avg = df.groupby('day')['avis_num'].mean()
    if daily_avg.empty:
        print("⚠️ Skipping daily average plot (no data).")
        return
    plt.figure(figsize=(10, 4))
    sns.lineplot(x=daily_avg.index, y=daily_avg.values, marker='o')
    plt.title('📅 Daily Average Satisfaction')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/daily_avg_satisfaction.png")
    plt.close()

def plot_center_average(df):
    center_avg = df.groupby('codeCentre')['avis_num'].mean().sort_values()
    if center_avg.empty:
        print("⚠️ Skipping center average plot (no data).")
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(x=center_avg.index, y=center_avg.values)
    plt.title('🏢 Average Score by Center')
    plt.ylabel('Average Score')
    plt.xlabel('Center')
    plt.tight_layout()
    plt.savefig("output/center_avg.png")
    plt.close()

def plot_distribution(df):
    if df.empty:
        print("⚠️ Skipping distribution plot (no data).")
        return
    plt.figure(figsize=(6, 4))
    sns.histplot(df['avis_num'], bins=4, kde=True)
    plt.title('📊 Satisfaction Score Distribution')
    plt.xlabel('Satisfaction Score')
    plt.tight_layout()
    plt.savefig("output/satisfaction_distribution.png")
    plt.close()

def plot_heatmap(df):
    if df.empty:
        print("⚠️ Skipping heatmap (no data).")
        return
    pivot_table = df.pivot_table(values='avis_num', index='codeCentre', columns='day', aggfunc='mean')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap='RdYlBu_r', cbar_kws={'label': 'Average Satisfaction'})
    plt.title("🔥 Average Satisfaction by Day and Center")
    plt.tight_layout()
    plt.savefig("output/heatmap_day_center.png")
    plt.close()

def plot_pie(df):
    if 'avis_num' not in df:
        print("⚠️ Skipping pie chart (no data).")
        return
    pie_data = df['avis_num'].map(reverse_map).value_counts()
    if pie_data.empty:
        print("⚠️ Skipping pie chart (no data).")
        return
    plt.figure(figsize=(6, 6))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
    plt.title("🧁 Satisfaction Categories Proportion")
    plt.tight_layout()
    plt.savefig("output/satisfaction_pie.png")
    plt.close()

def run_eda():
    df = fetch_evaluation_data()
    if df.empty:
        print("❌ No data returned from DB.")
        return
    df = preprocess(df)
    if df.empty:
        print("❌ No valid data after cleaning.")
        return
    plot_daily_average(df)
    plot_center_average(df)
    plot_distribution(df)
    plot_heatmap(df)
    plot_pie(df)
    print("✅ EDA complete — plots saved to /output folder.")

if __name__ == "__main__":
    run_eda()
