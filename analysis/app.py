# app.py
import joblib
from sklearn.preprocessing import OneHotEncoder
import base64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_connector import fetch_evaluation_data
import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="Satisfaction Dashboard", layout="wide")
# ---------- Preprocessing ----------
def preprocess(df):
    df['datetime'] = pd.to_datetime(df['dateEtHeure'])
    df['day'] = df['datetime'].dt.date
    return df

# ---------- Plot: Daily Average Line ----------
def plot_daily_average(df):
    daily_avg = df.groupby('day')['avis'].mean()
    fig, ax = plt.subplots()
    sns.lineplot(x=daily_avg.index, y=daily_avg.values, marker='o', ax=ax)
    ax.set_title("Daily Average Satisfaction")
    ax.set_ylabel("Average Score")
    ax.set_xlabel("Day")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ---------- Plot: Pie Chart ----------
def plot_pie(df):
    label_map = {
        5: 'Excellent',
        4: 'Très satisfait',
        3: 'Satisfait',
        2: 'Peu satisfait',
        1: 'Pas du tout satisfait'
}

    counts = df['avis'].value_counts().sort_index()
    labels = [label_map[val] for val in counts.index]

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title("Répartition des niveaux de satisfaction")
    ax.axis('equal')
    st.pyplot(fig)

# ---------- Plot: Heatmap (day × center) ----------
def plot_heatmap(df):
    pivot_table = df.pivot_table(index='day', columns='nom_centre', values='avis', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm_r', ax=ax)
    ax.set_title("Heatmap des scores moyens par jour et centre")
    st.pyplot(fig)

# ---------- download report ----------
def add_download_button():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(CURRENT_DIR, "..", "reports", "weekly_report.pdf")
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Télécharger le rapport hebdomadaire",
                data=f,
                file_name="rapport_hebdomadaire.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("⚠️ Rapport introuvable. Veuillez le générer d'abord.")


import joblib
from sklearn.preprocessing import OneHotEncoder


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'satisfaction_model.joblib'))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier du modèle est introuvable à ce chemin : {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict_satisfaction(df):
    # Create combined label like "Benguerir Si (center01)"
    df['center_display'] = df['nom_centre'] + " (" + df['codeCentre'] + ")"
    center_choice = st.selectbox("Select a Center", df['center_display'].unique())

    # Extract original values
    selected_row = df[df['center_display'] == center_choice].iloc[0]
    code = selected_row['codeCentre']
    name = selected_row['nom_centre']

    # Input for model
    input_df = pd.DataFrame({'codeCentre': [code], 'nom_centre': [name]})

    # Recreate encoder
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df[['codeCentre', 'nom_centre']])
    encoded_input = encoder.transform(input_df).toarray()

    # Predict
    prediction = model.predict(encoded_input)[0]
    st.success(f"✅ Predicted Satisfaction Score: **{prediction:.2f}** (out of 5)")


# ---------- Streamlit App ----------
def main():
    st.title("Tableau de bord de satisfaction - ADII")
    df = fetch_evaluation_data()

    if df.empty:
        st.warning("❌ Aucune donnée à afficher.")
        return

    df = preprocess(df)

    st.subheader(" Aperçu des données")
    st.dataframe(df.head(20))

    # 📊 All plots on one row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Moyenne quotidienne")
        plot_daily_average(df)

    with col2:
        st.subheader("Répartition satisfaction")
        plot_pie(df)

    with col3:
        st.subheader("Heatmap jour × centre")
        plot_heatmap(df)

    # 🔮 Prediction and 📄 Report in second row
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Prédiction")
        predict_satisfaction(df)

    with col5:
        st.subheader("Rapport hebdomadaire")
        add_download_button()


    

if __name__ == "__main__":
    main()
