import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data", layout="wide")

# Sidebar navigasi
menu = st.sidebar.radio("Pilih Menu", [
    "Klasifikasi Diabetes",
    "Pengelompokan Data",
    "Clustering Lokasi Gerai Kopi",
    "Visualisasi DBSCAN",
    "Input Lokasi Baru"
])

# Load model KNN dan scaler khusus diabetes
try:
    with open("model_knn.pkl", "rb") as f:
        model_knn = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        diabetes_scaler = pickle.load(f)
except FileNotFoundError:
    model_knn = None
    diabetes_scaler = None

# Load dataset kopi dan scaler lokal
df_kopi = pd.read_csv("lokasi_gerai_kopi_clean.csv")
X_kopi = df_kopi[["x", "y"]].values
kopi_scaler = StandardScaler()
X_scaled_kopi = kopi_scaler.fit_transform(X_kopi)
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
df_kopi["kmeans_label"] = kmeans.fit_predict(X_scaled_kopi)
agglo = AgglomerativeClustering(n_clusters=5)
df_kopi["agglo_label"] = agglo.fit_predict(X_scaled_kopi)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_kopi["dbscan_label"] = dbscan.fit_predict(X_scaled_kopi)

if menu == "Klasifikasi Diabetes":
    st.title("🔬 Klasifikasi Diabetes (KNN)")
    if model_knn is None or diabetes_scaler is None:
        st.warning("Model KNN atau scaler belum tersedia. Harap pastikan file model_knn.pkl dan scaler.pkl tersedia.")
    else:
        st.subheader("🔢 Masukkan Data Anda")
        pregnancies = st.number_input("Jumlah Kehamilan", 0, 20, step=1, value=0)
        glucose = st.number_input("Tingkat Glukosa", 0, 200, value=120)
        blood_pressure = st.number_input("Tekanan Darah", 0, 150, value=70)
        skin_thickness = st.number_input("Ketebalan Kulit", 0, 100, value=20)
        insulin = st.number_input("Tingkat Insulin", 0, 900, value=80)
        bmi = st.number_input("IMT (Indeks Massa Tubuh)", 0.0, 70.0, value=25.0)
        dpf = st.number_input("Fungsi Silsilah Diabetes (DPF)", 0.0, 3.0, value=0.5)
        age = st.number_input("Usia", 1, 120, step=1, value=30)

        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaled_input = diabetes_scaler.transform(input_data)

        if st.button("Prediksi Hasil"):
            prediksi = model_knn.predict(scaled_input)[0]
            hasil = "🟢 Tidak Diabetes" if prediksi == 0 else "🔴 Diabetes"
            st.success(f"Hasil Prediksi: **{hasil}**")
            if prediksi == 1:
                st.warning("Disarankan untuk berkonsultasi dengan profesional medis.")
            else:
                st.info("Risiko diabetes rendah berdasarkan data yang dimasukkan.")

elif menu == "Pengelompokan Data":
    st.title("📊 Pengelompokan Data dengan K-Means")
    st.markdown("""
    Nama : wahda 
    NIM  : 22146045
    """)

    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', header=None)
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                  'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = df.drop('Outcome', axis=1)
    data_scaled = StandardScaler().fit_transform(data)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(data_scaled)
    translated_cols = ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit', 
                       'Insulin', 'IMT', 'Fungsi Silsilah Diabetes', 'Usia', 'Hasil']
    df.columns = translated_cols
    df['Klaster'] = clusters

    st.subheader("📈 Visualisasi Pengelompokan (Glukosa vs IMT)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Glukosa', y='IMT', hue='Klaster', data=df, palette='viridis', ax=ax, s=100, alpha=0.7)
    plt.xlabel("Tingkat Glukosa")
    plt.ylabel("IMT (Indeks Massa Tubuh)")
    plt.title("Pengelompokan Data Diabetes berdasarkan Glukosa dan IMT")
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

elif menu == "Clustering Lokasi Gerai Kopi":
    st.title("📍 Clustering Lokasi Gerai Kopi")
    st.subheader("📌 KMeans (5 Klaster)")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=df_kopi["x"], y=df_kopi["y"], hue=df_kopi["kmeans_label"], palette="Set1", s=50, ax=ax1)
    ax1.set_title("KMeans Clustering")
    st.pyplot(fig1)

    st.subheader("🧬 Agglomerative Clustering (5 Klaster)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df_kopi["x"], y=df_kopi["y"], hue=df_kopi["agglo_label"], palette="tab10", s=50, ax=ax2)
    ax2.set_title("Agglomerative Clustering")
    st.pyplot(fig2)

elif menu == "Visualisasi DBSCAN":
    st.title("🌌 DBSCAN Clustering")
    fig3, ax3 = plt.subplots()
    for label in set(df_kopi["dbscan_label"]):
        cluster = df_kopi[df_kopi["dbscan_label"] == label]
        color = 'red' if label == -1 else None
        ax3.scatter(cluster["x"], cluster["y"], label=f"Klaster {label}" if label != -1 else "Noise", s=50, c=color)
    ax3.set_title("DBSCAN Clustering")
    ax3.legend()
    st.pyplot(fig3)

elif menu == "Input Lokasi Baru":
    st.title("🔢 Masukkan Lokasi Baru untuk Klasterisasi")
    x = st.number_input("Koordinat X", value=0.0)
    y = st.number_input("Koordinat Y", value=0.0)

    if st.button("Lihat Klaster"):
        input_data = np.array([[x, y]])
        scaled_input = kopi_scaler.transform(input_data)
        pred_kmeans = kmeans.predict(scaled_input)[0]
        st.success(f"📌 KMeans: Lokasi termasuk Klaster {pred_kmeans}")
        pred_agglo = agglo.fit_predict(np.vstack([X_scaled_kopi, scaled_input]))[-1]
        st.info(f"🧬 Agglomerative: Lokasi termasuk Klaster {pred_agglo}")
        pred_dbscan = dbscan.fit_predict(np.vstack([X_scaled_kopi, scaled_input]))[-1]
        st.warning(f"🌌 DBSCAN: Lokasi termasuk Klaster {pred_dbscan}" if pred_dbscan != -1 else "🌌 DBSCAN: Lokasi dianggap Noise")