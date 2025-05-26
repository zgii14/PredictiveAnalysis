# Laporan Proyek Machine Learning - Muhammad Rozagi

## Domain Proyek: Kesehatan
### Latar Belakang
Kualitas udara menjadi salah satu aspek penting dalam kehidupan masyarakat karena berdampak langsung terhadap kesehatan dan lingkungan. Peningkatan polusi udara akibat emisi industri, kendaraan bermotor, dan aktivitas manusia lainnya telah mendorong perlunya sistem prediksi yang mampu memberikan peringatan dini terhadap penurunan kualitas udara. Salah satu parameter utama yang digunakan untuk mengukur kualitas udara adalah konsentrasi partikel seperti PM2.5, PM10, NO₂, SO₂, dan CO.

Dalam beberapa tahun terakhir, pendekatan berbasis machine learning mulai banyak digunakan untuk memodelkan dan memprediksi kualitas udara. Muhaimin et al. (2023) menunjukkan bahwa metode Gradient Boosting Regression dapat memberikan hasil prediksi yang akurat terhadap konsentrasi PM2.5 di Kota Malang, menjadikannya alternatif yang menjanjikan dibanding metode konvensional. Sementara itu, Karyadi dan Santoso (2022) membandingkan performa model deep learning seperti LSTM, Bidirectional LSTM, dan GRU, dan menemukan bahwa pendekatan LSTM memberikan hasil yang cukup baik dalam memprediksi data time series kualitas udara.

Penggunaan algoritma seperti Random Forest, XGBoost, dan Support Vector Machine (SVM) juga telah banyak diteliti karena kemampuannya dalam menangani data non-linear dan kompleks. Sebagai contoh, Rahmadina et al. (2020) mengkaji klasifikasi kualitas udara di Jakarta menggunakan Random Forest dan membuktikan model tersebut efektif dalam mengklasifikasikan kategori udara secara tepat.

Dengan mempertimbangkan keberhasilan studi-studi sebelumnya, penelitian ini bertujuan membandingkan kinerja model Random Forest, XGBoost, dan SVM dalam memprediksi kategori kualitas udara berbasis data polutan dan lingkungan, sehingga dapat memberikan kontribusi dalam pengambilan keputusan mitigasi pencemaran udara.


### Referensi
Muhaimin, M. R., Karina, D. M., & Krisna, A. B. (2023). _Prediksi Kualitas Udara Malang Menggunakan Metode Gradient Boosting Regression_. Jurnal Digitech, 7(1).[https://jurnal.itscience.org/index.php/digitech/article/view/5046](https://jurnal.itscience.org/index.php/digitech/article/view/5046)

Karyadi, Y., & Santoso, H. (2022). _Prediksi Kualitas Udara Dengan Metoda LSTM, Bidirectional LSTM, dan GRU_. Jurnal JATISI, 9(1). [https://jurnal.mdp.ac.id/index.php/jatisi/article/view/1588](https://jurnal.mdp.ac.id/index.php/jatisi/article/view/1588)

Rahmadina, F., Nugroho, H. A., & Firdaus, A. (2020). _Klasifikasi Kualitas Udara Menggunakan Random Forest. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)_, 4(2), 327-334.[https://ejournal.undip.ac.id/index.php/resti/article/view/27444](https://ejournal.undip.ac.id/index.php/resti/article/view/27444)

## Business Understanding
### Problem Statements
- Bagaimana cara memprediksi kategori kualitas udara (seperti Good, Moderate, Poor) berdasarkan parameter lingkungan seperti suhu, kelembaban, PM2.5, PM10, NO₂, SO₂, CO, kedekatan dengan kawasan industri, dan kepadatan penduduk?
- Algoritma machine learning mana yang paling efektif dalam mengklasifikasikan kualitas udara dari berbagai daerah berdasarkan data polusi dan lingkungan?
- Fitur-fitur mana yang paling berpengaruh dalam menentukan kualitas udara menurut model machine learning?



### Goals
- Membangun model klasifikasi berbasis machine learning yang mampu memprediksi kategori kualitas udara dengan akurasi tinggi.
- Membandingkan performa berbagai algoritma machine learning seperti Random Forest, XGBoost, dan SVM dalam mengklasifikasikan kualitas udara.
- Mengidentifikasi fitur atau parameter lingkungan yang paling berkontribusi terhadap penentuan kualitas udara.

### Solution statements
- Menggunakan beberapa algoritma klasifikasi untuk membangun model prediksi kualitas udara, yaitu:
  - Random Forest Classifier  
  - XGBoost Classifier  
  - Support Vector Machine (SVM)  

- Melakukan serangkaian tahapan preprocessing data seperti:
  - Menghapus data duplikat dan membersihkan data tidak konsisten  
  - Menangani missing value (jika ada)  
  - Encoding label kategori kualitas udara menjadi numerik  
  - Standarisasi fitur numerik menggunakan `StandardScaler`  

- Melatih dan mengevaluasi model menggunakan metrik evaluasi klasifikasi:
  - Accuracy  
  - Recall  
  - F1-score  
  - Confusion Matrix  

- Membandingkan performa antar model untuk menentukan model klasifikasi terbaik berdasarkan metrik evaluasi

- Menguji model terbaik pada data baru untuk melihat generalisasi dan kemampuan prediksi terhadap data di luar pelatihan

- Menginterpretasikan hasil model dan pengaruh fitur terhadap klasifikasi kategori kualitas udara

## Data Understanding
Proyek ini menggunakan dataset Air Quality and Pollution Assessment dataset yang dapat diakses melalui Kaggle pada link berikut [Air Quality and Pollution Assessment dataset](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment/data).

## 📊 Variabel pada Dataset Prediksi Kualitas Udara

Berikut adalah deskripsi dari masing-masing fitur (variabel) dalam dataset `Air Quality and Pollution Assessment dataset` yang digunakan untuk memprediksi kualitas udara:

| Variabel                       | Deskripsi                                                                                     |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `Temperature`                 | Suhu udara lingkungan dalam satuan °C. Mempengaruhi reaksi kimia atmosferik dan konsentrasi polutan. |
| `Humidity`                   | Kelembaban relatif dalam persen (%). Berperan dalam pembentukan dan dispersi polutan udara.     |
| `PM2.5`                      | Partikulat halus berukuran ≤2.5 mikrometer. Partikel ini sangat berbahaya karena bisa masuk ke paru-paru dan aliran darah. |
| `PM10`                       | Partikulat kasar berukuran ≤10 mikrometer. Menyebabkan gangguan pernapasan dan iritasi saluran atas. |
| `NO2`                        | Konsentrasi Nitrogen Dioksida. Gas berbahaya dari pembakaran bahan bakar, berkontribusi pada polusi dan gangguan paru. |
| `SO2`                        | Konsentrasi Sulfur Dioksida. Gas beracun dari proses industri dan pembakaran batu bara.         |
| `CO`                         | Konsentrasi Karbon Monoksida. Gas beracun dari pembakaran tidak sempurna yang berbahaya bagi kesehatan. |
| `Proximity_to_Industrial_Areas` | Kedekatan terhadap kawasan industri. Semakin dekat, risiko paparan polutan lebih tinggi.         |
| `Population_Density`         | Kepadatan penduduk di suatu area. Semakin padat, umumnya tingkat polusi meningkat karena aktivitas manusia. |
| `Air_Quality`                | Label target klasifikasi. Kategori kualitas udara seperti: `Good`, `Moderate`, `Poor`, dll.    |

> Dataset ini digunakan untuk melatih model klasifikasi berbasis machine learning dalam memprediksi kategori kualitas udara berdasarkan parameter lingkungan dan polusi udara.


### Exploratory Data Analysis 
1. Informasi dataset
   <br>![Informasi dataset](img/df_info.png)
   - Ada 5000 baris dalam dataset.
   - Terdapat **10 kolom** fitur:
      1. `Temperature` (float64)
      2. `Humidity` (float64)
      3. `PM2.5` (float64)
      4. `PM10` (float64)
      5. `NO2` (float64)
      6. `SO2` (float64)
      7. `CO` (float64)
      8. `Proximity_to_Industrial_Areas` (float64)
      9. `Population_Density` (float64)
      10. `Air Quality` (object - target variabel)
    - Tidak terdapat nilai kosong (semua kolom memiliki 5000 non-null entries).
2. Deskripsi statistik fitur numerik dataset
   <br>![statistik](img/describe.png)
   <br>Berdasarkan deskripsi statistik, dataset ini memiliki 8 fitur numerik dengan sebaran nilai yang cukup bervariasi. Rata-rata suhu berada di kisaran 30°C, dengan kelembapan sekitar 70%. Konsentrasi PM2.5 dan PM10 menunjukkan nilai maksimum yang cukup tinggi, mengindikasikan potensi outlier. Nilai-nilai untuk gas pencemar seperti NO2, SO2, dan CO juga menunjukkan variasi yang signifikan. Fitur Proximity to Industrial Areas dan Population Density memiliki distribusi yang relatif normal namun tetap menunjukkan rentang nilai yang lebar, menandakan keragaman kondisi lingkungan dalam dataset.

3. Visualisasi Kategori Udara 
<br>![statistik](img/Distribusi_Kategori.png)
**Insight:**
- Grafik distribusi menunjukkan bahwa sebagian besar kualitas udara berada pada kategori Good dan Moderate, sementara Poor dan Hazardous relatif sedikit. Ketidakseimbangan ini penting diperhatikan, terutama jika data akan digunakan untuk klasifikasi, karena bisa memengaruhi kinerja model. Temuan ini juga membuka peluang analisis lanjutan berdasarkan waktu atau lokasi.
Visualisasi boxplot menunjukkan adanya outlier pada variabel pm10 (242), so2 (46), co (248), o3 (188), dan no2 (198), namun outlier ini masih berada dalam rentang normal berdasarkan distribusi data, sehingga tidak perlu dihapus. Untuk analisis selanjutnya, akan digunakan model Random Forest (RF) dan XGBoost, yang dikenal robust terhadap outlier, sehingga potensi dampak dari nilai ekstrem dalam dataset dapat diminimalkan secara alami oleh model.

![dist](https://github.com/user-attachments/assets/5430adf7-7d1e-44b0-97b1-5a99528fce5b)

Gambar tersebut merupakan grafik visualisasi dari distribusi kolom kategori pada dataset. Pada sumbu horizontal (x-axis), terdapat kategori kualitas udara, yaitu ‘SEDANG’, ‘BAIK’, ‘TIDAK SEHAT’, ‘SANGAT TIDAK SEHAT’, ‘BERBAHAYA’, dan ‘TIDAK ADA DATA’. Sementara itu, sumbu vertikal (y-axis) menunjukkan jumlah kemunculan masing-masing kategori dalam 
dataset. Dari grafik, dapat disimpulkan bahwa: 
1. Kategori sedang memiliki jumlah data tertinggi. 
2. Kategori tidak sehat merupakan kategori kedua terbanyak. 
3. Kategori baik, sangat tidak sehat, dan berbahaya memiliki jumlah data 
yang jauh lebih rendah. 
4. Kategori tidak ada data mencerminkan adanya data yang tidak memiliki 
informasi terkait kategori kualitas udara. 
5. Kategori tidak ada data dan berbahaya memiliki jumlah data yang sangat 
sedikit. Sehingga diputuskan untuk menghapus kategori berbahaya.

![corr](https://github.com/user-attachments/assets/c881bf1c-df3f-46de-a198-02fd30df1f28)

Gambar yang ditampilkan di atas adalah heatmap korelasi antar polutan udara, yang menunjukkan hubungan linear antara setiap pasangan variabel polutan seperti pm2.5, pm10, so2, co, o3, dan no2. Nilai korelasi berkisar dari -1 hingga 1, di mana:

1. Nilai mendekati 1 menunjukkan korelasi positif kuat (jika satu naik, yang lain juga cenderung naik),
2. Nilai mendekati -1 menunjukkan korelasi negatif kuat (jika satu naik, yang lain cenderung turun),
3. Nilai mendekati 0 menunjukkan korelasi lemah atau tidak ada korelasi linear.

   
## Data Preparation
Pada bagian ini akan dilakukan 3 tahap persiapan data, yaitu:
1. Encoding Fitur Kategori
   <br>Mengubah data kategorik menjadi numerik agar bisa diproses oleh algoritma machine learning, dalam kasus ini menggunakan One-Hot Encoding dan Label Encoding. Karena algoritma machine learning pada umumnya hanya dapat memproses data numerik, sehingga untuk memudahkan proses pemodelan, data kategorik harus diencoding.
   <br>![image](img/encoder.png)

2. Train-Test-Split
   <br>Dataset dibagi menjadi data latih (train) dan data uji (test) menggunakan train_test_split dari sklearn dengan rasio 80:20. Hal ini dilakukan untuk memisahkan data pada proses pelatihan dan evaluasi model.
   <br>![image](img/split.png)
3. Standarisasi
   <br>Proses penskalaan fitur numerik agar berada dalam rentang yang seragam menggunakan StandardScaler. Hal ini bertujuan agar model bekerja adil dan optimal terhadap semua fitur, serta tidak ada bias skala.
   <br>![image](img/standarisasi.png)

## Modeling
Tahapan ini bertujuan membangun model machine learning untuk memprediksi status diabetes (positif/negatif) berdasarkan fitur kesehatan yang tersedia. Pada proyek ini, saya membangun lima model machine learning, yaitu:
1. Logistic Regression
```python
# Model Logistic
logistic = LogisticRegression(class_weight='balanced', max_iter=1000)
logistic.fit(X_train,y_train)
```
- Tahapan:
  - Inisialisasi model Logistic Regression.
  - Menyeimbangkan kelas target agar model tidak bias terhadap kelas mayoritas.
  - Melatih model menggunakan data train.
- Parameter:
  - class_weight='balanced': menghitung bobot kelas secara otomatis berdasarkan distribusi kelas.
  - max_iter=1000: untuk menentukan jumlah maksimum iterasi agar konvergen.

2. K-Nearest Neighbors (KNN)
```python
# Model KNN
knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='distance', metric = 'minkowski')
knn.fit(X_train,y_train)
```
- Tahapan:
  - Inisialisasi model KNN.
  - Menentukan parameter jumlah tetangga dan metode pembobotan.
  - Melatih model dengan menyimpan data latih.
- Parameter:
  - n_neighbors=5: menggunakan 5 tetangga terdekat.
  - p=2: menggunakan jarak Euclidean (karena p=2).
  - weights='distance': tetangga yang lebih dekat memiliki bobot lebih besar.
  - metric='minkowski': metode pengukuran jarak umum, digunakan bersama p.

3. Support Vector Classifier (SVC)
```python
# Model SVC
svc = SVC(kernel = 'rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state = 42)
svc.fit(X_train,y_train)
```
- Tahapan:
  - Inisialisasi model SVC.
  - Menentukan kernel dan parameter regulasi.
  - Melatih model untuk memisahkan kelas dengan margin maksimum.
- Parameter:
  - kernel='rbf': menggunakan fungsi kernel radial basis (non-linear).
  - C=1.0: Parameter regulasi untuk mengontrol overfitting.
  - gamma='scale': parameter kernel otomatis berdasarkan data.
  - class_weight='balanced': menyesuaikan bobot kelas minoritas.
  - probability=True: mengaktifkan prediksi probabilitas.
  - random_state=42: menjamin hasil yang konsisten.

4. Decision Tree
```python
# Model Decision Tree
decisionTree = DecisionTreeClassifier(criterion= 'entropy', class_weight='balanced', random_state=42)
decisionTree.fit(X_train,y_train)
```
- Tahapan:
  - Inisialisasi model decision tree.
  - Menentukan strategi pemisahan node menggunakan entropy.
  - Melatih model berdasarkan data pelatihan.
- Parameter:
  - criterion='entropy': menggunakan informasi gain untuk membagi node.
  - class_weight='balanced': untuk menyesuaikan bobot kelas minoritas.
  - random_state=42: agar hasil selalu konsisten.

5. Random Forest
```python
# Model Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion= 'entropy', class_weight='balanced', random_state=42)
rf.fit(X_train,y_train)
```
- Tahapan:
  - Inisialisasi model
  - Menentukan jumlah pohon dan kriteria pemisahan.
  - Melatih model dengan membangun banyak decision tree.
- Parameter:
  - n_estimators=10: jumlah pohon dalam hutan.
  - criterion='entropy': untuk membagi node.
  - class_weight='balanced': untuk menyeimbangkan kontribusi tiap kelas.
  - random_state=42: untuk replikasi hasil.


**Kelebihan dan kekurangan dari setiap algoritma yang digunakan**
| **Algoritma**                       | **Kelebihan**                                                                                                                                           | **Kekurangan**                                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**             | - Sederhana dan cepat dalam pelatihan<br>- Hasilnya mudah diinterpretasikan<br>- Cocok untuk prediksi probabilistik                                     | - Kurang optimal untuk hubungan non-linear<br>- Rentan terhadap multikolinearitas antar fitur                           |
| **Support Vector Classifier (SVC)** | - Efektif untuk data berdimensi tinggi<br>- Mampu memisahkan kelas dengan margin maksimal<br>- Mendukung kernel non-linear                              | - Sensitif terhadap pemilihan parameter (C, gamma)<br>- Waktu komputasi tinggi untuk data besar                         |
| **K-Nearest Neighbors (KNN)**       | - Mudah dipahami dan diimplementasikan<br>- Tidak memerlukan proses pelatihan (lazy learning)<br>- Non-parametrik (tidak mengasumsikan distribusi data) | - Lambat pada dataset besar karena menghitung jarak setiap kali prediksi<br>- Sensitif terhadap outlier dan skala fitur |
| **Decision Tree**                   | - Mudah dibaca dan divisualisasikan<br>- Tidak memerlukan normalisasi fitur<br>- Dapat menangani data numerik dan kategorik                             | - Mudah overfitting jika tidak dilakukan pruning<br>- Tidak stabil terhadap perubahan kecil pada data                   |
| **Random Forest**                   | - Lebih akurat dibanding satu pohon (ensembling)<br>- Mengurangi overfitting melalui agregasi<br>- Tahan terhadap noise dan outlier                     | - Interpretabilitas lebih rendah dibanding decision tree<br>- Membutuhkan waktu dan resource lebih banyak               |


**Model Terbaik: Random Forest**
<br>Random Forest dipilih sebagai model terbaik karena memberikan hasil evaluasi paling tinggi dan seimbang pada data uji, tahan terhadap overfitting, serta mampu menangani data tidak seimbang dengan baik. Kombinasi antara akurasi tinggi dan metrik klasifikasi yang kuat menjadikannya pilihan yang optimal untuk menyelesaikan permasalahan klasifikasi ini. Untuk lebih jelasnya, dapat dilihat pada Evaluation.

**Melakukan Hyperparameter Tuning pada Model Terbaik**
```python
# Grid parameter yang akan diuji
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Setup GridSearch
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
```
Hyperparameter tuning bertujuan untuk mencari kombinasi parameter terbaik agar performa model (model terbaik: Random Forest) meningkat, terutama pada metrik F1-score karena:
- Data bersifat tidak seimbang (kelas minoritas penting)
- F1-score mempertimbangkan keseimbangan antara precision dan recall
<br>Metode yang digunakan adalah GridSearchCV, yang mana digunakan untuk melakukan pencarian kombinasi hyperparameter terbaik secara eksploratif melalui pencarian grid (grid search) dengan cross-validation (cv=5).

| Hyperparameter      | Deskripsi                                                               |
| ------------------- | ----------------------------------------------------------------------- |
| `n_estimators`      | Jumlah pohon dalam forest. Dicoba nilai 100 dan 200.                     |
| `max_depth`         | Kedalaman maksimal pohon. Dicoba `None` (bebas) dan 20.                 |
| `min_samples_split` | Minimum jumlah data yang dibutuhkan untuk membagi node. Dicoba 2 dan 5. |
| `min_samples_leaf`  | Minimum jumlah data di setiap daun. Dicoba 1 dan 2.                     |
| `max_features`      | Jumlah fitur yang dipertimbangkan saat membagi. Dicoba `'sqrt'`.        |


## Evaluation
**Metrik yang Digunakan**
1. Accuracy, yaitu persentase prediksi yang benar terhadap seluruh data.
   <br>Formula:
   
   $$
     \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   $$
   
   <br>Keterangan:
   - TP = True Positive (Data positif yang diprediksi benar sebagai positif)
   - TN = True Negative (Data negatif yang diprediksi benar sebagai negatif)
   - FP = False Positive (Data negatif yang salah diprediksi sebagai positif)
   - FN = False Negative (Data positif yang salah diprediksi sebagai negatif)

2. Precision, yaitu proporsi data yang diprediksi positif yang benar-benar positif. Metrik ini cocok digunakan saat false positive harus diminimalkan.
   <br>Formula:<br>
   
$$
  \text{Precision} = \frac{TP}{TP + FP}
$$
   
3. Recall (Sensitivity), yaitu roporsi data positif yang berhasil dikenali model. Penting jika false negative berisiko tinggi, seperti pada kasus deteksi penyakit.
   <br>Formula:<br>
   
$$
  \text{Recall} = \frac{TP}{TP + FN}
$$

4. F1-Score, yaitu harmonic mean dari Precision dan Recall. Digunakan saat perlu keseimbangan antara Precision dan Recall.

   <br>Formula:<br>
   
$$
  \text{F1 Score} = 2 \times  \frac{\text{Precision} \times  \text{Recall}}{\text{Precision} + \text{Recall}}
$$

5. ROC-AUC (Receiver Operating Characteristic - Area Under Curve), yaitu luas area di bawah kurva ROC, yang menggambarkan trade-off antara True Positive Rate (TPR) dan False Positive Rate (FPR).
   - Nilai maksimal adalah 1 (semakin mendekati 1 semakin baik).
   - Tidak bergantung pada threshold tertentu.

**Hasil Evaluasi**
| **Metrik**    | **LogisticRegression** | **KNN**  | **SVC**  | **DecisionTree** | **RandomForest** | **RandomForest (setelah tuning)** |
| ------------- | ---------------------- | -------- | -------- | ---------------- | ---------------- | --------------------------------- |
| **Accuracy**  | 0.845229               | 0.959068 | 0.843869 | 0.953966         | 0.969783         | 0.970747                          |
| **Precision** | 0.237569               | 0.688312 | 0.242612 | 0.560170         | 0.840278         | 0.88                              |
| **Recall**    | 0.883243               | 0.401081 | 0.931892 | 0.568649         | 0.523243         | 0.51                              |
| **F1-score**  | 0.374427               | 0.506831 | 0.384993 | 0.564378         | 0.644903         | 0.65                              |
| **ROC-AUC**   | 0.942800               | 0.831969 | 0.943392 | 0.772177         | 0.884897         | 0.947724                          |

| Model                  | Train Accuracy | Test Accuracy |
| ---------------------- | -------------- | ------------- |
| Logistic Regression    | 0.845187       | 0.845229      |
| KNN                    | 0.999164       | 0.959068      |
| SVC                    | 0.844889       | 0.843869      |
| Decision Tree          | 0.999164       | 0.953966      |
| Random Forest          | 0.994416       | 0.969783      |
| Random Forest (Tuning) | 0.999164       | 0.970747      |


**Visualisasi**
- Confusion Matrix Kelima Model yang Dibangun
  <br>![conf_matrix](img/conf_matrix.png)
- Perbandingan Akurasi pada Setiap Model
  <br>![perbandingan_akurasi](img/perbandingan_akurasi.png)
- Perbandingan Metrik pada Setiap Model
  <br>![perbandingan_metrik](img/perbandingan_metrik.png)
- Perbandingan Hasil Prediksi dan Nilai Aktual (Randomm Forest - setelah tuning)
  <br>![image](img/hasilpred_aktual.png)
- Feature Importance Berdasarkan Random Forest (setelah tuning)
  <br>![feature_important](img/feature_important.png)


<br>Berdasarkan hasil evaluasi menggunakan metrik-metrik seperti accuracy, precision, recall, F1-score, dan ROC-AUC, dapat disimpulkan bahwa model Random Forest terutama setelah dilakukan hyperparameter tuning merupakan model terbaik secara keseluruhan untuk kasus ini. Hal ini didasarkan pada poin-poin berikut:
1. Performa Evaluasi Terbaik:
   - Model Random Forest (setelah tuning) memperoleh skor tertinggi pada sebagian besar metrik penting seperti accuracy (97.07%), precision (88%), F1-score (0.65), dan ROC-AUC (0.947).
   - Meskipun recall-nya (0.51) lebih rendah dibandingkan model Logistic Regression (0.88) dan SVC (0.93), trade-off ini masih dapat diterima, terutama karena precision meningkat secara signifikan. Hal ini berarti model lebih baik dalam menghindari false positives, yang mungkin penting dalam konteks data ini tergantung pada problem statement.
2. Trade-off yang Seimbang:
   - Dalam konteks klasifikasi yang memiliki ketimpangan kelas (class imbalance), F1-score dan ROC-AUC menjadi metrik yang lebih representatif dibandingkan hanya mengandalkan accuracy.
   - Random Forest (setelah tuning) memberikan keseimbangan terbaik antara precision dan recall, ditunjukkan dari F1-score tertinggi (0.65), yang merepresentasikan kompromi optimal antara keduanya.
3. Performa Training dan Testing:
   - Tuning pada Random Forest meningkatkan test accuracy dari 96.97% menjadi 97.07%, menunjukkan bahwa tuning memberikan peningkatan meskipun tidak terlalu besar.
   - Meskipun train accuracy-nya sangat tinggi (99.91%), yang menunjukkan adanya sedikit overfitting, namun selisih train-test accuracy masih dalam batas yang dapat diterima, dan generalisasi model masih tergolong baik.
4. Model Lain sebagai Pembanding:
   - KNN dan Decision Tree menunjukkan performa yang tinggi di training set namun mengalami penurunan di test set, mengindikasikan overfitting yang lebih parah.
   - Logistic Regression dan SVC memiliki recall tinggi namun precision dan F1-score-nya sangat rendah, yang kurang ideal jika false positive berdampak signifikan dalam konteks penggunaan model.

Kesimpulan akhir:
<br>Model Random Forest dengan hyperparameter tuning adalah pilihan paling tepat sebagai model terbaik, karena memberikan keseimbangan performa terbaik dan hasil evaluasi yang unggul secara umum. Pemilihan metrik seperti F1-score dan ROC-AUC yang lebih representatif daripada sekadar akurasi sangat penting dalam konteks ini, terutama jika data tidak seimbang atau konsekuensi kesalahan klasifikasi berbeda antara kelas.

## Conclusion
1. Menjawab Problem Statement
   - Model klasifikasi yang dibangun berhasil memprediksi status diabetes seseorang (positif atau negatif) berdasarkan variabel-variabel kesehatan seperti usia, jenis kelamin, BMI, kadar HbA1c, kadar glukosa darah, riwayat hipertensi dan penyakit jantung, serta kebiasaan merokok.
   - Dari hasil evaluasi performa lima algoritma klasifikasi, Random Forest Classifier terbukti menjadi model terbaik terutama setelah dilakukan proses hyperparameter tuning. Model ini memberikan hasil prediksi yang seimbang dan akurat, bahkan pada kelas minoritas (positif diabetes), dengan ROC-AUC mencapai 0.9477.
   - Selain itu, melalui analisis feature importance dari model Random Forest, dapat disimpulkan bahwa HbA1c_level, blood_glucose_level, gender, dan bmi merupakan faktor yang paling berpengaruh terhadap kemungkinan seseorang mengidap diabetes.

2. Mencapai Goals
   <br>Proyek ini berhasil mengembangkan sistem klasifikasi diabetes berbasis machine learning yang:
    - Mampu mengklasifikasikan status diabetes secara akurat menggunakan data medis.
    - Membandingkan performa lima algoritma klasifikasi secara objektif menggunakan metrik evaluasi seperti accuracy, precision, recall, F1-score, dan ROC-AUC.
    - Menentukan algoritma optimal (Random Forest) melalui hyperparameter tuning untuk meningkatkan performa deteksi.
    - Memberikan insight terhadap fitur yang paling berkontribusi dalam prediksi diabetes, sehingga hasil dapat digunakan sebagai rekomendasi pendukung keputusan medis.

4. Dampak Solusi Statement
   <br>Solusi yang dibangun bersifat aplikatif dan siap digunakan sebagai sistem bantu deteksi dini diabetes berdasarkan data kesehatan sederhana. Dengan performa tinggi pada metrik klasifikasi, solusi ini dapat membantu pihak medis atau instansi kesehatan dalam penapisan awal risiko diabetes, memprioritaskan pasien berisiko tinggi untuk penanganan lebih lanjut.
Penggunaan Random Forest yang tahan terhadap outlier dan mampu menangani fitur kompleks menjadikan model ini pilihan yang andalan dan dapat diandalkan dalam konteks nyata.
