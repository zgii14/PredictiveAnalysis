# -*- coding: utf-8 -*-
"""Submission_Akhir

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15h-VVy6mq4JFZo4R0ZlHAJD99wyFWKK2

# ***Import Library* yang akan digunakan**

Tahap awal dalam proses analisis ini dimulai dengan melakukan import library yang diperlukan. Library seperti pandas dan numpy digunakan untuk manipulasi dan analisis data, sementara matplotlib.pyplot dan seaborn digunakan untuk visualisasi. Modul dari scikit-learn digunakan untuk preprocessing, pembuatan pipeline, pemodelan, dan evaluasi performa model. Selain itu, digunakan juga XGBoost sebagai salah satu algoritma klasifikasi. warnings.filterwarnings('ignore') digunakan untuk menyembunyikan peringatan agar output tetap bersih, dan np.random.seed(42) ditetapkan untuk memastikan hasil yang konsisten pada setiap eksekusi.
"""

# === Library Dasar ===
import numpy as np
import pandas as pd

# === Visualisasi ===
import matplotlib.pyplot as plt
import seaborn as sns

# === Preprocessing dan Pipeline ===
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# === Model Selection dan Evaluasi ===
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score
)

# === Algoritma Pembelajaran Mesin ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# === Pengaturan Umum ===
import warnings
warnings.filterwarnings('ignore')

# === Reproducibility ===
np.random.seed(42)

"""# **Upload Dataset melalui Kaggle**

Kode pada bagian ini adalah serangkaian perintah yang digunakan di lingkungan Google Colab untuk mengunggah file kredensial Kaggle, mengunduh dataset dari Kaggle, mengekstrak file dataset, dan memuat serta menampilkan informasi dasar tentang dataset tersebut.
"""

# Import module yang disediakan google colab untuk kebutuhan upload file

from google.colab import files
files.upload()

"""Untuk mengakses data dari Kaggle, pertama dibuat direktori .kaggle jika belum ada. Kemudian, file kaggle.json disalin ke direktori tersebut dan diatur izin aksesnya agar hanya bisa dibaca oleh pemilik, guna menjaga keamanan kredensial."""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

"""Perintah !kaggle datasets download -d mujtabamatin/air-quality-and-pollution-assessment digunakan untuk mengunduh dataset Air Quality and Pollution Assessment dari Kaggle menggunakan API. Setelah autentikasi berhasil, perintah ini akan menyimpan file dataset dalam format zip ke direktori kerja saat ini."""

!kaggle datasets download -d mujtabamatin/air-quality-and-pollution-assessment

"""Perintah !unzip -q air-quality-and-pollution-assessment -d aqindex digunakan untuk mengekstrak file dataset yang telah diunduh sebelumnya. File ZIP akan diekstrak secara diam-diam (-q untuk quiet mode) ke dalam folder bernama aqindex."""

!unzip -q air-quality-and-pollution-assessment -d aqindex

"""Perintah df = pd.read_csv('/content/aqindex/updated_pollution_dataset.csv') digunakan untuk membaca file dataset hasil ekstraksi ke dalam bentuk DataFrame menggunakan pandas. File updated_pollution_dataset.csv berada di dalam folder aqindex, dan hasil pembacaannya disimpan dalam variabel df untuk proses analisis lebih lanjut."""

df=pd.read_csv('/content/aqindex/updated_pollution_dataset.csv')

"""Perintah df.info() digunakan untuk menampilkan informasi umum tentang struktur DataFrame, seperti jumlah baris dan kolom, nama kolom, tipe data di setiap kolom, serta jumlah nilai non-null (tidak hilang) pada masing-masing kolom."""

df.info()

"""Perintah df.describe() digunakan untuk menampilkan statistik deskriptif dari kolom numerik dalam DataFrame, seperti mean, standar deviasi, nilai minimum, maksimum, serta kuartil (25%, 50%, 75%), yang membantu memahami sebaran data secara umum."""

df.describe()

"""Perintah df.head(5) digunakan untuk menampilkan lima baris pertama dari DataFrame. Ini berguna untuk mendapatkan gambaran awal tentang struktur dan isi data, termasuk nama kolom dan contoh nilai di setiap kolom."""

df.head(5)

"""# **Exploratory Data Analysis**

Pada tahap eksplorasi data ini, dilakukan analisis distribusi kategori kualitas udara menggunakan visualisasi diagram batang. Visualisasi ini bertujuan untuk melihat seberapa seimbang data pada setiap kategori, yaitu Good, Moderate, Poor, dan Hazardous. Grafik menunjukkan bahwa kategori Good merupakan yang paling dominan dalam dataset, diikuti oleh Moderate, Poor, dan Hazardous. Hal ini mengindikasikan bahwa sebagian besar pengamatan kualitas udara berada dalam kondisi baik hingga sedang, sementara kondisi yang buruk dan berbahaya jumlahnya relatif lebih sedikit. Visualisasi ini dibuat dengan menggunakan fungsi countplot dari library Seaborn, yang menampilkan frekuensi kemunculan masing-masing kategori dalam bentuk batang vertikal. Untuk memperjelas tampilan, ditambahkan elemen-elemen seperti judul grafik, label sumbu, rotasi teks pada sumbu-x, dan garis bantu (grid) horizontal pada sumbu-y. Analisis ini menjadi langkah awal yang penting untuk memahami persebaran kualitas udara dalam dataset sebelum masuk ke tahap pemodelan atau analisis lebih lanjut.
"""

plt.figure(figsize=(10, 6))
sns.countplot(x='Air Quality', data=df, palette='viridis')
plt.title('Distribusi Kategori Kualitas Udara')
plt.xlabel('Kategori Kualitas Udara')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

"""**Insight:**
- Grafik distribusi menunjukkan bahwa sebagian besar kualitas udara berada pada kategori Good dan Moderate, sementara Poor dan Hazardous relatif sedikit. Ketidakseimbangan ini penting diperhatikan, terutama jika data akan digunakan untuk klasifikasi, karena bisa memengaruhi kinerja model. Temuan ini juga membuka peluang analisis lanjutan berdasarkan waktu atau lokasi.

Pada kode dibawah, dilakukan visualisasi distribusi untuk setiap fitur numerik dalam dataset menggunakan histogram. Fitur-fitur yang dianalisis meliputi Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO, Proximity to Industrial Areas, dan Population Density. Histogram ini membantu memahami pola distribusi masing-masing variabel, apakah menyebar normal, skewed, atau memiliki outlier. Selain itu, grafik ini juga memberi gambaran awal mengenai variabilitas dan konsentrasi data pada setiap fitur, yang penting untuk proses prapemrosesan dan pemodelan selanjutnya.
"""

# Daftar kolom numerik
numerical_cols = [
    'Temperature',
    'Humidity',
    'PM2.5',
    'PM10',
    'NO2',
    'SO2',
    'CO',
    'Proximity_to_Industrial_Areas',
    'Population_Density'
]

# Plot histogram untuk setiap fitur numerik
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribusi {col.replace("(", "").replace(")", "").replace("-", " ").title()}')
    plt.xlabel(f'{col.replace("(", "").replace(")", "").replace("-", " ").title()}')
    plt.ylabel('Frekuensi')
    plt.grid(True)
    plt.show()

"""**Insight:**
- Berdasarkan visualisasi distribusi fitur numerik, sebagian besar variabel seperti PM2.5, PM10, SO2, dan CO menunjukkan sebaran yang condong ke kanan (right-skewed), menandakan adanya nilai ekstrem tinggi. Sementara itu, Temperature, Humidity, NO2, dan Population Density memiliki distribusi yang cenderung normal atau mendekatinya. Fitur Proximity to Industrial Areas tampak memiliki distribusi bimodal, yang mengindikasikan kemungkinan adanya dua kelompok wilayah dengan jarak yang berbeda terhadap kawasan industri. Insight ini penting untuk pertimbangan transformasi data dan deteksi outlier sebelum pemodelan.

Pada kode dibawah dilakukan Visualisasi matriks korelasi untuk menunjukkan hubungan antar fitur numerik dalam dataset. Beberapa variabel tampak memiliki korelasi yang cukup kuat, baik positif maupun negatif.
"""

plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.show()

"""**Insight:**
- Berdasarkan matriks korelasi, terlihat bahwa terdapat hubungan yang sangat kuat antara PM2.5 dan PM10, menunjukkan bahwa kedua polutan ini cenderung meningkat bersamaan, kemungkinan berasal dari sumber emisi yang sama. CO juga memiliki korelasi yang cukup tinggi dengan NO2 dan SO2,

# ***Data Cleaning***

Sebelum masuk ke tahap analisis lebih lanjut, dilakukan proses data cleaning untuk memastikan kualitas data yang digunakan. Tahap ini mencakup identifikasi dan penanganan data yang hilang (missing values), duplikasi data, serta deteksi nilai-nilai yang tidak wajar atau outlier yang dapat mempengaruhi hasil analisis. Selain itu, dilakukan juga pengecekan tipe data untuk memastikan kesesuaian antara tipe data dan nilai yang dikandungnya. Langkah-langkah ini penting untuk menjamin bahwa data yang digunakan dalam pemodelan atau visualisasi selanjutnya bersih, konsisten, dan siap untuk diolah.

Kode dibawah untuk mengecek apakah ada data yang kosong perkolom
"""

print(df.isnull().sum())

"""Kode dibawah untuk mengecek apakah ada data yang duplikasi"""

duplicate_count = df.duplicated().sum()
print("\nJumlah baris duplikat:", duplicate_count)

"""Kode dibawah merupakan kode untuk cek outlier"""

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

print("\nJumlah Outlier per Kolom:")
for col in numerical_cols:
    outlier_count = detect_outliers(df, col)
    print(f"{col}: {outlier_count} outlier")

"""**Insight:**
- Dari analisis outlier pada data kualitas udara, ditemukan nilai ekstrem terutama pada variabel PM2.5, PM10, dan NO2, yang menunjukkan kemungkinan adanya lonjakan polusi pada waktu atau lokasi tertentu. Outlier juga muncul pada variabel lingkungan seperti Temperature dan Humidity, yang bisa mencerminkan kondisi cuaca ekstrem. Dalam konteks kualitas udara, outlier ini tidak selalu menunjukkan kesalahan, melainkan bisa menjadi indikasi kejadian penting dan sesudah tadi melihat di bagian EDA jadi diputuskan untuk tidak dihapus berhubungan model yang dibangun juga cukup robust terhadap outlier
"""

print(df)

"""# ***Data Split***

Setelah melalui tahap pembersihan data (data cleaning) yang mencakup penanganan missing values dan deteksi outlier, langkah selanjutnya adalah mempersiapkan data untuk proses pelatihan model. Pada tahap ini, dilakukan pemisahan antara fitur dan target variabel. Kolom 'Air Quality' dipilih sebagai target karena merepresentasikan kualitas udara yang ingin diprediksi. Sementara itu, seluruh kolom lainnya digunakan sebagai fitur atau variabel independen yang menjadi input bagi model. Pada kode dibawah juga terdapat encoding terhadap fitur target agar menjadi numerik supaya dapat diproses oleh algoritma
"""

X = df.drop('Air Quality', axis=1)
y = df['Air Quality']

# Label encoding untuk target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

"""X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) = Membagi data menjadi set latih (80%) dan uji (20%) dengan seed acak 42"""

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

"""- print("Ukuran X_train:", X_train.shape) = Menampilkan dimensi set fitur latih

- print("Ukuran X_test:", X_test.shape) = Menampilkan dimensi set fitur uji

- print("Ukuran y_train:", y_train.shape) = Menampilkan dimensi set label latih

- print("Ukuran y_test:", y_test.shape) = Menampilkan dimensi set label uji
"""

print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test:", X_test.shape)
print("Ukuran y_train:", y_train.shape)
print("Ukuran y_test:", y_test.shape)

"""# **Modeling dengan Random Forest, XGBoost dan SVM**

Setelah data selesai diproses dan siap digunakan, tahap selanjutnya adalah pemodelan menggunakan tiga algoritma klasifikasi: Random Forest, XGBoost, dan Support Vector Machine (SVM). Masing-masing model dibangun dalam bentuk pipeline yang mencakup proses standardisasi data menggunakan StandardScaler, diikuti oleh algoritma klasifikasi yang sesuai. Penggunaan pipeline bertujuan untuk menyederhanakan alur preprocessing dan training model, sekaligus memastikan proses yang konsisten dan efisien selama evaluasi dan pengujian.
"""

# === 2. BUAT PIPELINE UNTUK MASING-MASING MODEL ===

# a. Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# b. XGBoost
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        n_estimators=600,
        max_depth=15,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    ))
])

# c. SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

"""## **Evaluasi**

Setelah model dilatih, tahap berikutnya adalah melakukan evaluasi performa untuk menilai seberapa baik masing-masing model dalam memprediksi kualitas udara. Proses evaluasi dilakukan menggunakan fungsi evaluate_model, yang menghitung metrik seperti accuracy, F1 score, dan recall. Selain itu, ditampilkan juga classification report serta confusion matrix dalam bentuk visual untuk memberikan gambaran detail mengenai performa model terhadap setiap kelas target. Evaluasi ini dilakukan secara konsisten untuk semua pipeline model yang telah dibangun sebelumnya.
"""

# === 3. FUNGSI UNTUK EVALUASI MODEL ===

def evaluate_model(pipeline, model_name):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"\n=== {model_name} ===")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Recall:", recall)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='rocket_r',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

"""Pada kode dibawah dilakukan pemanggilan fungsi evaluasi yang sudah kita buat diatas"""

evaluate_model(rf_pipeline, "Random Forest")
evaluate_model(xgb_pipeline, "XGBoost")
evaluate_model(svm_pipeline, "Support Vector Machine (SVM)")

from sklearn.metrics import precision_score, roc_auc_score

# Prediksi
y_pred_rf = rf_pipeline.fit(X_train, y_train).predict(X_test)
y_pred_xgb = xgb_pipeline.fit(X_train, y_train).predict(X_test)
y_pred_svm = svm_pipeline.fit(X_train, y_train).predict(X_test)

# Inisialisasi tabel evaluasi
model_scores = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                            columns=['RandomForest', 'XGBoost', 'SVM'])

# Random Forest
model_scores.loc['accuracy', 'RandomForest'] = accuracy_score(y_test, y_pred_rf)
model_scores.loc['precision', 'RandomForest'] = precision_score(y_test, y_pred_rf, average='weighted')
model_scores.loc['recall', 'RandomForest'] = recall_score(y_test, y_pred_rf, average='weighted')
model_scores.loc['f1_score', 'RandomForest'] = f1_score(y_test, y_pred_rf, average='weighted')
model_scores.loc['roc_auc', 'RandomForest'] = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test), multi_class='ovr', average='weighted')

# XGBoost
model_scores.loc['accuracy', 'XGBoost'] = accuracy_score(y_test, y_pred_xgb)
model_scores.loc['precision', 'XGBoost'] = precision_score(y_test, y_pred_xgb, average='weighted')
model_scores.loc['recall', 'XGBoost'] = recall_score(y_test, y_pred_xgb, average='weighted')
model_scores.loc['f1_score', 'XGBoost'] = f1_score(y_test, y_pred_xgb, average='weighted')
model_scores.loc['roc_auc', 'XGBoost'] = roc_auc_score(y_test, xgb_pipeline.predict_proba(X_test), multi_class='ovr', average='weighted')

# SVM
model_scores.loc['accuracy', 'SVM'] = accuracy_score(y_test, y_pred_svm)
model_scores.loc['precision', 'SVM'] = precision_score(y_test, y_pred_svm, average='weighted')
model_scores.loc['recall', 'SVM'] = recall_score(y_test, y_pred_svm, average='weighted')
model_scores.loc['f1_score', 'SVM'] = f1_score(y_test, y_pred_svm, average='weighted')
model_scores.loc['roc_auc', 'SVM'] = roc_auc_score(y_test, svm_pipeline.predict_proba(X_test), multi_class='ovr', average='weighted')

# Tampilkan hasil akhir
print("Hasil Evaluasi Model Klasifikasi:")
display(model_scores)

"""## Inferensi

Setelah model dievaluasi dan menunjukkan performa yang baik, dilakukan tahap inferensi untuk menguji kemampuan model dalam memprediksi kualitas udara pada data baru. Tiga sampel data (test_samples) dengan nilai parameter lingkungan yang berbeda diuji menggunakan ketiga model: Random Forest, XGBoost, dan SVM. Setiap model memberikan prediksi kategori kualitas udara berdasarkan fitur-fitur seperti suhu, kelembapan, konsentrasi polutan (PM2.5, PM10, NO2, SO2, CO), kedekatan dengan area industri, dan kepadatan penduduk. Hasil prediksi kemudian dikembalikan ke bentuk label asli untuk interpretasi yang lebih mudah.
"""

# === TEST DATA BARU ===
test_samples = [
    [30.2, 62.0, 7.1, 18.0, 22.0, 10.0, 1.70, 6.2, 400],
    [24.5, 55.0, 4.0, 10.0, 10.5, 4.5, 1.00, 12.0, 280],
    [33.0, 80.0, 55.0, 100.0, 45.0, 20.0, 2.50, 3.5, 850],
]
feature_names = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
x_test_new = pd.DataFrame(test_samples, columns=feature_names)

# === PREDIKSI DENGAN RANDOM FOREST ===
pred_rf = rf_pipeline.predict(x_test_new)
pred_rf_labels = le.inverse_transform(pred_rf)
print("=== Random Forest Predictions ===")
for i, label in enumerate(pred_rf_labels):
    print(f"Data uji ke-{i+1}: {x_test_new.iloc[i].to_dict()}")
    print(f"Prediksi Kategori Kualitas Udara (RF): {label}\n")

# === PREDIKSI DENGAN XGBOOST ===
pred_xgb = xgb_pipeline.predict(x_test_new)
pred_xgb_labels = le.inverse_transform(pred_xgb)
print("=== XGBoost Predictions ===")
for i, label in enumerate(pred_xgb_labels):
    print(f"Data uji ke-{i+1}: {x_test_new.iloc[i].to_dict()}")
    print(f"Prediksi Kategori Kualitas Udara (XGB): {label}\n")

# === PREDIKSI DENGAN SVM ===
pred_svm = svm_pipeline.predict(x_test_new)
pred_svm_labels = le.inverse_transform(pred_svm)
print("=== SVM Predictions ===")
for i, label in enumerate(pred_svm_labels):
    print(f"Data uji ke-{i+1}: {x_test_new.iloc[i].to_dict()}")
    print(f"Prediksi Kategori Kualitas Udara (SVM): {label}\n")