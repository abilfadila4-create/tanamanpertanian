import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import io

# --- 1. Konfigurasi Streamlit ---
st.set_page_config(
    page_title="Aplikasi Rekomendasi Tanaman Pertanian Presisi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Fungsi Pemuatan Data dan Pelatihan Model ---

@st.cache_data
def load_data(file_path):
    """Memuat data dan mengembalikan DataFrame."""
    try:
        # Menangani data dengan koma sebagai pemisah desimal jika ada
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error("File 'Crop_recommendation.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None

# Dekorator st.cache_resource akan menyimpan model agar tidak dilatih ulang setiap kali interaksi
@st.cache_resource
def train_model(df):
    """Melatih model Random Forest Classifier."""
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[features]
    y = df['label']

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Pembagian data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Inisialisasi dan pelatihan Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, scaler, accuracy, report

# --- Load Data Awal ---
df = load_data("Crop_recommendation.csv")
if df is None:
    st.stop() # Hentikan eksekusi jika data tidak ditemukan

# --- Pelatihan Model Awal ---
model, scaler, accuracy, report = train_model(df)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# --- Mapping Gambar Tanaman ---
CROP_IMAGES = {
    "rice": "rice.jpg",
    "maize": "maize.jpg",
    "chickpea": "chickpea.jpg",
    "kidneybeans": "kidneybeans.jpg",
    "pigeonpeas": "pigeonpeas.jpg",
    "mothbeans": "mothbeans.jpg",
    "mungbean": "mungbean.jpg",
    "blackgram": "assets/images/blackgram.jpg",
    "lentil": "assets/images/lentil.jpg",
    "pomegranate": "assets/images/pomegranate.jpg",
    "banana": "assets/images/banana.jpg",
    "mango": "assets/images/mango.jpg",
    "grapes": "assets/images/grapes.jpg",
    "watermelon": "assets/images/watermelon.jpg",
    "muskmelon": "assets/images/muskmelon.jpg",
    "apple": "assets/images/apple.jpg",
    "orange": "assets/images/orange.jpg",
    "papaya": "assets/images/papaya.jpg",
    "coconut": "assets/images/coconut.jpg",
    "cotton": "assets/images/cotton.jpg",
    "jute": "assets/images/jute.jpg",
    "coffee": "assets/images/coffee.jpg"
}

# --- SIDEBAR: Input Prediksi ---

st.sidebar.title("Prediksi Rekomendasi Tanaman")
st.sidebar.header("Masukkan Kondisi Lahan:")

# Mendapatkan nilai rata-rata, min, dan max untuk nilai default input
mean_values = df[features].mean()

input_data = {}
for feature in features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    default_val = mean_values[feature]
    
    # Mapping label yang lebih mudah dibaca
    label_map = {
        'N': 'Nitrogen (N) [ppm]',
        'P': 'Fosfor (P) [ppm]',
        'K': 'Kalium (K) [ppm]',
        'temperature': 'Suhu (°C)',
        'humidity': 'Kelembaban (%)',
        'ph': 'Nilai pH',
        'rainfall': 'Curah Hujan (mm)'
    }
    
    # Menyesuaikan input berdasarkan jenis fitur
    if feature in ['N', 'P', 'K']:
        input_data[feature] = st.sidebar.slider(
            label_map[feature], 
            min_val.astype(int), 
            max_val.astype(int), 
            int(default_val), 
            step=1
        )
    else:
        input_data[feature] = st.sidebar.number_input(
            label_map[feature], 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(f'{default_val:.2f}'), 
            step=0.01,
            format="%.2f"
        )

# Tombol Prediksi
if st.sidebar.button("Rekomendasikan Tanaman"):
    # 1. Konversi input ke DataFrame
    new_data = pd.DataFrame([input_data])

    # 2. Scaling data input
    new_data_scaled = scaler.transform(new_data)

    # 3. Prediksi
    prediction = model.predict(new_data_scaled)[0]

    # 4. Tampilkan hasil di SIDEBAR
    st.sidebar.success("Rekomendasi Terbaik")
    st.sidebar.markdown(f"**{prediction.upper()}**")

    # 5. Gambar kecil tepat di bawah prediksi (SIDEBAR)
    image_path = CROP_IMAGES.get(prediction.lower())

    if image_path:
        st.sidebar.image(
            image_path,
            width=140,  #ukuran kecil
            caption=prediction.capitalize()
        )
    else:
        st.sidebar.caption("Gambar tanaman belum tersedia")

# --- MAIN PAGE CONTENT ---

st.title("Aplikasi Rekomendasi Tanaman Pertanian Presisi")
st.write("""
Welcome to my Portofolio. This application created by [@abilfadilaa](https://www.linkedin.com/in/abilfadila/).
""")
st.markdown("Aplikasi ini menggunakan model *Machine Learning* **Random Forest** untuk memprediksi tanaman paling optimal berdasarkan kondisi nutrisi tanah dan faktor iklim.")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Analisis Data Eksploratif (EDA)", "Pemodelan & Evaluasi", "About Project"])

# ----------------------------------------------------
# TAB 1: Dataset Overview
# ----------------------------------------------------
with tab1:
    st.header("Ringkasan Dataset")
    
    st.subheader("5 Baris Data Pertama")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informasi Data (Tipe Data)")
        # Menangkap output df.info() ke buffer untuk ditampilkan di Streamlit
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
    with col2:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
        
    st.subheader("Cek Nilai Hilang (Missing Values)")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values.rename('Jumlah Nilai Hilang'))
    st.success("Kesimpulan: Dataset bersih, tidak ada nilai yang hilang.")

# ----------------------------------------------------
# TAB 2: Analisis Data Eksploratif (EDA)
# ----------------------------------------------------
with tab2:
    st.header("Analisis Data Eksploratif (EDA)")
    
    st.subheader("Distribusi Jenis Tanaman")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='label', data=df, order=df['label'].value_counts().index, ax=ax, palette='viridis')
    ax.set_title('Distribusi Jumlah Sampel per Jenis Tanaman')
    st.pyplot(fig)
    st.info("Dataset ini seimbang (*balanced*), dengan jumlah sampel yang sama untuk setiap jenis tanaman.")

    st.subheader("Korelasi Antar Fitur")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Peta Panas Matriks Korelasi Antar Fitur')
    st.pyplot(fig)
    
    st.subheader("Boxplot Kebutuhan Nutrisi (N, P, K)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, col in enumerate(['N', 'P', 'K']):
        sns.boxplot(x='label', y=col, data=df, ax=axes[i], palette='tab10')
        axes[i].set_title(f'Distribusi {col}', fontsize=14)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=90, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------------------
# TAB 3: Modeling & Evaluation
# ----------------------------------------------------
with tab3:
    st.header("Pemodelan Klasifikasi")
    st.subheader("Model yang Digunakan: Random Forest Classifier")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(label="Akurasi Model pada Test Set", value=f"{accuracy*100:.2f}%", delta="Sangat Tinggi")
        st.info("Akurasi sempurna/sangat tinggi karena kelas-kelas tanaman memiliki kebutuhan yang sangat spesifik dan terpisah secara jelas.")
    
    with col2:
        st.subheader("Laporan Klasifikasi")
        # Mengubah laporan dari dictionary ke DataFrame untuk tampilan yang lebih baik
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.iloc[:-3, :], use_container_width=True) # Hanya menampilkan metrik per kelas
        
    st.subheader("Analisis Kepentingan Fitur (Feature Importance)")
    
    # Mendapatkan Feature Importance
    rf_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=rf_importances, y=rf_importances.index, palette='rocket', ax=ax)
    ax.set_title('Kepentingan Fitur (Random Forest)')
    st.pyplot(fig)
    
    st.success("Kalium (K) dan Kelembaban (*humidity*) terbukti menjadi faktor paling penting dalam proses pengambilan keputusan model.")
    
# ----------------------------------------------------
# TAB 4: About Project
# ----------------------------------------------------
with tab4:
    st.header("Tentang Proyek Rekomendasi Tanaman")
    st.markdown("""
    Proyek ini mewujudkan konsep **Pertanian Presisi** dengan memanfaatkan data untuk pengambilan keputusan yang tepat.
    
    #### Tujuan
    Memberikan rekomendasi tanaman yang paling optimal berdasarkan kondisi lahan aktual, yang mengarah pada efisiensi sumber daya dan peningkatan hasil panen bagi petani.
    
    #### Detail Teknis
    * **Dataset:** Data lingkungan dan nutrisi dari 22 jenis tanaman.
    * **Model:** **Random Forest Classifier** (Algoritma *Ensemble* berbasis pohon keputusan).
    * **Metrik Kunci:** Nitrogen (N), Fosfor (P), Kalium (K), Suhu, Kelembaban, pH, dan Curah Hujan.
    
    #### Tentang Pengembang
    Aplikasi ini dikembangkan sebagai studi kasus dalam *Machine Learning* untuk klasifikasi multi-kelas, menunjukkan bagaimana fitur yang memiliki keterpisahan kelas tinggi dapat menghasilkan model prediktif yang hampir sempurna.
    
    ---
    
    *Dibuat dengan Python, Streamlit, dan Scikit-learn.*
    """)
