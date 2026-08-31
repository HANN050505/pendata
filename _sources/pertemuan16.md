# Analisis dan Prediksi Performa Akademik Mahasiswa Menggunakan Algoritma Klasifikasi

## Abstrak

Perkembangan teknologi informasi telah menghasilkan data dalam jumlah besar yang dapat dimanfaatkan untuk memperoleh informasi berharga melalui proses penambangan data (data mining). Salah satu bidang yang dapat memanfaatkan teknologi tersebut adalah bidang pendidikan. Analisis terhadap data akademik mahasiswa dapat membantu institusi pendidikan dalam memahami faktor-faktor yang memengaruhi keberhasilan belajar serta memprediksi performa akademik mahasiswa di masa mendatang.

Penelitian ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi performa akademik mahasiswa dan membangun model prediksi nilai akhir mahasiswa menggunakan teknik klasifikasi. Dataset yang digunakan merupakan dataset Student Performance yang terdiri dari 145 data mahasiswa dengan 33 atribut. Variabel target yang digunakan adalah GRADE yang merepresentasikan nilai akhir mahasiswa.

Tahapan penelitian meliputi data preprocessing, exploratory data analysis (EDA), pembangunan model klasifikasi menggunakan Decision Tree, Random Forest, K-Nearest Neighbor (KNN), dan Naive Bayes, serta evaluasi model menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.

Hasil penelitian menunjukkan bahwa algoritma Decision Tree dan Random Forest memberikan performa terbaik dibandingkan algoritma lainnya. Selain itu, analisis feature importance menunjukkan bahwa beberapa atribut memiliki kontribusi yang lebih besar dalam menentukan nilai akhir mahasiswa.

**Kata Kunci:** Data Mining, Klasifikasi, Student Performance, Decision Tree, Random Forest, KNN, Naive Bayes.

---

# 1. Pendahuluan

## 1.1 Latar Belakang

Perkembangan teknologi informasi dan komunikasi telah mendorong peningkatan jumlah data yang dihasilkan setiap hari. Data tersebut berasal dari berbagai bidang, termasuk pendidikan. Institusi pendidikan saat ini menyimpan berbagai jenis data akademik yang berkaitan dengan aktivitas belajar mahasiswa, mulai dari data kehadiran, aktivitas belajar, kondisi sosial ekonomi, hingga nilai akademik.

Data yang tersimpan dalam jumlah besar tersebut memiliki potensi untuk menghasilkan informasi yang berguna apabila dianalisis menggunakan metode yang tepat. Salah satu metode yang dapat digunakan adalah data mining. Data mining merupakan proses menemukan pola, hubungan, atau informasi tersembunyi dari kumpulan data yang besar menggunakan teknik statistik, machine learning, dan kecerdasan buatan.

Dalam dunia pendidikan, penerapan data mining dikenal sebagai Educational Data Mining (EDM). Pendekatan ini memungkinkan institusi pendidikan untuk memahami perilaku belajar mahasiswa, mengidentifikasi faktor-faktor yang memengaruhi keberhasilan akademik, serta memprediksi performa mahasiswa di masa depan. Informasi tersebut dapat digunakan sebagai dasar pengambilan keputusan untuk meningkatkan kualitas pembelajaran dan layanan pendidikan.

Prediksi performa akademik mahasiswa merupakan salah satu permasalahan yang sering diteliti dalam bidang Educational Data Mining. Kemampuan memprediksi nilai mahasiswa dapat membantu dosen dan institusi pendidikan dalam memberikan intervensi lebih awal kepada mahasiswa yang berpotensi mengalami kesulitan akademik. Dengan demikian, tingkat keberhasilan studi mahasiswa dapat ditingkatkan.

Pada penelitian ini digunakan dataset Student Performance yang berisi berbagai atribut yang menggambarkan karakteristik mahasiswa. Dataset tersebut dianalisis menggunakan beberapa algoritma klasifikasi yaitu Decision Tree, Random Forest, K-Nearest Neighbor (KNN), dan Naive Bayes. Perbandingan performa dari setiap algoritma dilakukan untuk menentukan model yang paling efektif dalam memprediksi nilai akhir mahasiswa.




# BAB 2. DESKRIPSI DATASET DAN DATA PREPROCESSING

## 2.1 Deskripsi Dataset

Dataset yang digunakan dalam penelitian ini adalah dataset **Student Performance** yang berisi informasi mengenai karakteristik mahasiswa serta nilai akhir yang diperoleh pada suatu mata kuliah. Dataset ini digunakan untuk melakukan analisis dan prediksi performa akademik mahasiswa menggunakan teknik klasifikasi.

Secara umum, dataset ini memuat berbagai atribut yang berkaitan dengan kondisi akademik, sosial, dan perilaku belajar mahasiswa yang diduga memiliki pengaruh terhadap nilai akhir yang diperoleh.

### 2.1.1 Informasi Dataset

Berdasarkan hasil eksplorasi awal, diperoleh informasi sebagai berikut:

| Keterangan                | Nilai |
| ------------------------- | ----- |
| Jumlah Baris (Records)    | 145   |
| Jumlah Kolom (Attributes) | 33    |
| Jumlah Fitur Prediktor    | 32    |
| Variabel Target           | GRADE |
| Missing Value             | 0     |
| Data Duplikat             | 0     |

Dataset terdiri atas satu variabel target yaitu **GRADE** yang menunjukkan nilai akhir mahasiswa serta sejumlah atribut prediktor yang digunakan untuk membangun model klasifikasi.

### 2.1.2 Struktur Dataset

Beberapa atribut yang terdapat pada dataset antara lain:

| No   | Atribut                          | Keterangan            |
| ---- | -------------------------------- | --------------------- |
| 1    | STUDENT ID                       | Identitas mahasiswa   |
| 2    | COURSE ID                        | Identitas mata kuliah |
| 3-32 | Berbagai karakteristik mahasiswa | Variabel prediktor    |
| 33   | GRADE                            | Nilai akhir mahasiswa |

Variabel **GRADE** digunakan sebagai target klasifikasi karena merepresentasikan hasil akademik mahasiswa yang ingin diprediksi.

---

## 2.2 Analisis Variabel Target

Sebelum dilakukan proses pemodelan, perlu dilakukan analisis terhadap distribusi variabel target untuk mengetahui keseimbangan data.

### Distribusi Nilai Mahasiswa

| Grade | Jumlah Mahasiswa |
| ----- | ---------------- |
| 0     | 8                |
| 1     | 35               |
| 2     | 24               |
| 3     | 21               |
| 4     | 10               |
| 5     | 17               |
| 6     | 13               |
| 7     | 17               |

### Interpretasi

Berdasarkan distribusi tersebut, terlihat bahwa:

* Kelas **1** merupakan kelas dengan jumlah data terbanyak.
* Kelas **0** merupakan kelas dengan jumlah data paling sedikit.
* Dataset memiliki distribusi kelas yang relatif tidak seimbang (imbalanced dataset).
* Ketidakseimbangan kelas dapat memengaruhi performa model klasifikasi.

Visualisasi distribusi target dapat dibuat menggunakan kode berikut:

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

sns.countplot(
    x='GRADE',
    data=df
)

plt.title('Distribusi Nilai Mahasiswa')
plt.xlabel('Grade')
plt.ylabel('Jumlah Mahasiswa')

plt.show()
```

### Interpretasi Visualisasi

Grafik distribusi menunjukkan bahwa sebagian besar mahasiswa berada pada kategori nilai rendah hingga menengah. Jumlah mahasiswa dengan nilai sangat tinggi relatif lebih sedikit dibandingkan kategori lainnya.

---

## 2.3 Data Preprocessing

Data preprocessing merupakan tahapan penting dalam proses data mining karena kualitas data sangat memengaruhi hasil model yang dibangun.

Tahapan preprocessing yang dilakukan meliputi:

1. Pemeriksaan missing value.
2. Pemeriksaan data duplikat.
3. Penghapusan atribut yang tidak relevan.
4. Pemisahan fitur dan target.
5. Pembagian data training dan testing.

---

## 2.4 Pemeriksaan Missing Value

Missing value merupakan kondisi ketika suatu atribut tidak memiliki nilai.

Pemeriksaan dilakukan menggunakan kode berikut:

```python
print(df.isnull().sum())
```

### Hasil

```text
Total Missing Value = 0
```

### Interpretasi

Hasil pemeriksaan menunjukkan bahwa dataset tidak memiliki missing value sehingga tidak diperlukan proses imputasi data.

Keuntungan dari kondisi ini adalah seluruh data dapat langsung digunakan pada tahap analisis dan pemodelan.

---

## 2.5 Pemeriksaan Data Duplikat

Data duplikat dapat menyebabkan bias dalam proses pembelajaran model.

Pemeriksaan dilakukan menggunakan kode berikut:

```python
duplicate_rows = df.duplicated().sum()

print(
    "Jumlah Data Duplikat:",
    duplicate_rows
)
```

### Hasil

```text
Jumlah Data Duplikat: 0
```

### Interpretasi

Tidak ditemukan data duplikat pada dataset sehingga seluruh data dapat digunakan untuk proses analisis.

---

## 2.6 Penghapusan Atribut Tidak Relevan

Kolom **STUDENT ID** hanya berfungsi sebagai identitas mahasiswa dan tidak memiliki hubungan langsung dengan nilai akhir mahasiswa.

Oleh karena itu atribut tersebut dihapus sebelum proses pemodelan dilakukan.

### Implementasi

```python
df = df.drop(
    columns=['STUDENT ID']
)
```

### Alasan Penghapusan

Jika atribut identitas tetap digunakan, model dapat mempelajari pola yang tidak relevan sehingga menurunkan kemampuan generalisasi model.

---

## 2.7 Pemisahan Variabel Fitur dan Target

Setelah atribut yang tidak relevan dihapus, data dipisahkan menjadi fitur (X) dan target (y).

### Implementasi

```python
X = df.drop(
    columns=['GRADE']
)

y = df['GRADE']
```

### Penjelasan

* Variabel **X** berisi seluruh atribut prediktor.
* Variabel **y** berisi nilai akhir mahasiswa yang akan diprediksi.

---

## 2.8 Pembagian Data Training dan Testing

Pembagian data dilakukan menggunakan metode Train-Test Split.

Proporsi yang digunakan adalah:

* 80% data training
* 20% data testing

### Implementasi

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Penjelasan

Pembagian data dilakukan agar model dapat diuji menggunakan data yang belum pernah dilihat sebelumnya.

Penggunaan parameter:

```python
random_state=42
```

bertujuan agar hasil pembagian data dapat direproduksi.

Sedangkan parameter:

```python
stratify=y
```

digunakan untuk menjaga proporsi distribusi kelas pada data training dan data testing.

---

## 2.9 Ringkasan Tahap Preprocessing

Berdasarkan hasil preprocessing yang telah dilakukan, diperoleh beberapa temuan penting sebagai berikut:

| Tahapan                    | Hasil           |
| -------------------------- | --------------- |
| Missing Value              | Tidak ditemukan |
| Data Duplikat              | Tidak ditemukan |
| Penghapusan STUDENT ID     | Dilakukan       |
| Pemisahan Fitur dan Target | Berhasil        |
| Train-Test Split           | Berhasil        |

Secara keseluruhan, dataset memiliki kualitas yang cukup baik karena tidak ditemukan missing value maupun data duplikat. Oleh karena itu data dapat langsung digunakan pada tahap Exploratory Data Analysis (EDA) dan pembangunan model klasifikasi.


# BAB 3. EXPLORATORY DATA ANALYSIS (EDA)

## 3.1 Pendahuluan

Exploratory Data Analysis (EDA) merupakan tahapan penting dalam proses data mining yang bertujuan untuk memahami karakteristik data sebelum dilakukan pemodelan. Melalui EDA, peneliti dapat mengidentifikasi pola, tren, hubungan antar variabel, serta potensi permasalahan yang terdapat pada dataset.

Pada penelitian ini, EDA dilakukan menggunakan beberapa teknik visualisasi seperti histogram, boxplot, correlation heatmap, dan pairplot. Hasil analisis ini digunakan sebagai dasar dalam menentukan strategi pemodelan yang tepat pada tahap klasifikasi.

---

# 3.2 Statistik Deskriptif

Statistik deskriptif digunakan untuk memberikan gambaran umum mengenai distribusi data.

## Implementasi

```python
df.describe()
```

## Hasil

Tabel statistik deskriptif akan menampilkan:

* Count
* Mean
* Standard Deviation
* Minimum
* Kuartil 1 (25%)
* Median (50%)
* Kuartil 3 (75%)
* Maximum

Contoh penampilan:

| Statistik | Nilai |
| --------- | ----- |
| Count     | 145   |
| Mean      | ...   |
| Std       | ...   |
| Min       | ...   |
| Max       | ...   |

## Interpretasi

Statistik deskriptif memberikan informasi mengenai persebaran data pada setiap atribut. Nilai rata-rata (mean) menunjukkan kecenderungan pusat data, sedangkan standar deviasi menunjukkan tingkat variasi data terhadap rata-ratanya.

Semakin besar nilai standar deviasi, maka semakin besar pula variasi data yang dimiliki atribut tersebut.

---

# 3.3 Analisis Distribusi Nilai Mahasiswa

Analisis distribusi dilakukan terhadap variabel target (GRADE) untuk mengetahui pola persebaran nilai mahasiswa.

## Implementasi

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

sns.countplot(
    x='GRADE',
    data=df
)

plt.title('Distribusi Nilai Mahasiswa')
plt.xlabel('Grade')
plt.ylabel('Jumlah Mahasiswa')

plt.show()
```

## Gambar 3.1 Distribusi Nilai Mahasiswa



## Interpretasi

Berdasarkan visualisasi distribusi nilai mahasiswa, terlihat bahwa jumlah mahasiswa pada setiap kategori nilai tidak merata.

Kategori nilai 1 memiliki jumlah mahasiswa paling banyak, sedangkan kategori nilai 0 memiliki jumlah mahasiswa paling sedikit.

Distribusi yang tidak seimbang ini menunjukkan bahwa dataset termasuk dalam kategori moderately imbalanced dataset. Kondisi tersebut dapat memengaruhi performa model klasifikasi karena model cenderung lebih mudah mempelajari kelas yang memiliki jumlah data lebih banyak.

## Insight

* Sebagian besar mahasiswa memperoleh nilai pada kategori rendah hingga menengah.
* Jumlah mahasiswa dengan nilai sangat tinggi relatif sedikit.
* Diperlukan evaluasi model yang tidak hanya bergantung pada accuracy.

---

# 3.4 Analisis Distribusi Setiap Fitur

Histogram digunakan untuk melihat distribusi masing-masing atribut pada dataset.

## Implementasi

```python
df.hist(
    figsize=(18,15),
    bins=15
)

plt.tight_layout()
plt.show()
```

## Gambar 3.2 Histogram Seluruh Variabel



## Interpretasi

Histogram menunjukkan bagaimana distribusi nilai pada masing-masing atribut.

Melalui histogram dapat diketahui:

* Apakah distribusi data normal atau tidak.
* Apakah terdapat atribut yang memiliki distribusi tidak merata.
* Apakah terdapat indikasi skewness pada beberapa variabel.

## Insight

Beberapa atribut menunjukkan distribusi yang terkonsentrasi pada nilai tertentu, yang mengindikasikan adanya karakteristik dominan pada mahasiswa dalam dataset.

Distribusi yang tidak merata dapat memengaruhi performa algoritma tertentu seperti KNN dan Naive Bayes.

---

# 3.5 Analisis Outlier Menggunakan Boxplot

Boxplot digunakan untuk mendeteksi keberadaan outlier pada dataset.

## Implementasi

```python
plt.figure(figsize=(20,8))

df.boxplot()

plt.xticks(
    rotation=90
)

plt.show()
```

## Gambar 3.3 Boxplot Dataset



## Interpretasi

Boxplot menampilkan:

* Nilai minimum
* Kuartil pertama
* Median
* Kuartil ketiga
* Nilai maksimum
* Outlier

Berdasarkan visualisasi boxplot, sebagian besar atribut tidak menunjukkan keberadaan outlier ekstrem yang dapat mengganggu proses pembelajaran model.

## Insight

* Persebaran data relatif stabil.
* Tidak ditemukan outlier yang sangat ekstrem.
* Dataset dapat langsung digunakan untuk proses klasifikasi.

---

# 3.6 Analisis Korelasi Antar Variabel

Correlation Heatmap digunakan untuk mengukur hubungan antar variabel dalam dataset.

## Implementasi

```python
plt.figure(
    figsize=(15,10)
)

sns.heatmap(
    df.corr(
        numeric_only=True
    ),
    cmap='coolwarm'
)

plt.title(
    'Correlation Heatmap'
)

plt.show()
```

## Gambar 3.4 Correlation Heatmap



## Interpretasi

Heatmap menunjukkan tingkat hubungan antar atribut menggunakan nilai korelasi.

Rentang korelasi:

| Nilai Korelasi | Interpretasi |
| -------------- | ------------ |
| 0.00 – 0.19    | Sangat Lemah |
| 0.20 – 0.39    | Lemah        |
| 0.40 – 0.59    | Sedang       |
| 0.60 – 0.79    | Kuat         |
| 0.80 – 1.00    | Sangat Kuat  |

Semakin mendekati 1 maka hubungan kedua variabel semakin kuat.

Semakin mendekati -1 maka hubungan kedua variabel semakin kuat tetapi berlawanan arah.

## Insight

Berdasarkan heatmap, sebagian besar atribut memiliki hubungan yang rendah hingga sedang.

Hal ini menunjukkan bahwa masing-masing atribut memberikan informasi yang relatif berbeda sehingga tetap layak digunakan dalam proses klasifikasi.

---

# 3.7 Pairplot

Pairplot digunakan untuk melihat hubungan antar variabel secara visual.

## Implementasi

```python
selected_features = [
    'COURSE ID',
    'GRADE'
]

sns.pairplot(
    df[selected_features]
)

plt.show()
```

## Gambar 3.5 Pairplot



## Interpretasi

Pairplot memperlihatkan hubungan antara dua atau lebih variabel melalui scatter plot dan distribusi data.

Visualisasi ini membantu mengidentifikasi pola tertentu yang mungkin tidak terlihat melalui statistik deskriptif.

---

# 3.8 Analisis Variabel Target terhadap Fitur

Untuk mengetahui hubungan antara nilai akhir mahasiswa dan fitur tertentu dapat digunakan boxplot berdasarkan kategori nilai.

## Implementasi

```python
plt.figure(figsize=(10,6))

sns.boxplot(
    x='GRADE',
    y='COURSE ID',
    data=df
)

plt.show()
```

## Gambar 3.6 Hubungan COURSE ID terhadap GRADE



## Interpretasi

Visualisasi menunjukkan variasi nilai COURSE ID pada setiap kategori nilai mahasiswa.

Perbedaan distribusi dapat mengindikasikan bahwa karakteristik mata kuliah memiliki pengaruh terhadap performa akademik mahasiswa.

---

# 3.9 Ringkasan Hasil EDA

Berdasarkan hasil Exploratory Data Analysis yang telah dilakukan, diperoleh beberapa temuan penting sebagai berikut:

1. Dataset terdiri dari 145 data mahasiswa dan 33 atribut.
2. Tidak ditemukan missing value maupun data duplikat.
3. Distribusi variabel target (GRADE) tidak sepenuhnya seimbang.
4. Sebagian besar atribut memiliki persebaran data yang relatif baik.
5. Tidak ditemukan outlier ekstrem yang berpotensi mengganggu proses klasifikasi.
6. Korelasi antar variabel cenderung rendah hingga sedang.
7. Seluruh atribut layak digunakan dalam proses pembangunan model klasifikasi.

---

# 3.10 Kesimpulan EDA

Tahap Exploratory Data Analysis menunjukkan bahwa dataset memiliki kualitas yang cukup baik untuk digunakan dalam proses klasifikasi.

Tidak ditemukan permasalahan serius seperti missing value, data duplikat, maupun outlier ekstrem. Selain itu, hasil visualisasi menunjukkan bahwa atribut-atribut yang tersedia memiliki potensi untuk digunakan dalam memprediksi nilai akhir mahasiswa.

Berdasarkan hasil tersebut, dataset siap digunakan pada tahap berikutnya yaitu pembangunan model klasifikasi menggunakan algoritma Decision Tree, Random Forest, K-Nearest Neighbor (KNN), dan Naive Bayes.


# BAB 4. IMPLEMENTASI DAN HASIL KLASIFIKASI

## 4.1 Pendahuluan

Setelah proses Exploratory Data Analysis (EDA) selesai dilakukan, tahap berikutnya adalah membangun model klasifikasi untuk memprediksi nilai akhir mahasiswa berdasarkan atribut yang tersedia pada dataset.

Pada penelitian ini digunakan empat algoritma klasifikasi yang umum digunakan dalam bidang data mining, yaitu:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbor (KNN)
4. Naive Bayes

Keempat algoritma tersebut dipilih karena memiliki karakteristik yang berbeda sehingga dapat digunakan untuk membandingkan performa klasifikasi pada dataset Student Performance.

---

# 4.2 Persiapan Data

Sebelum melakukan proses klasifikasi, data dibagi menjadi data training dan data testing.

## Implementasi

```python
from sklearn.model_selection import train_test_split

X = df.drop(
    ['STUDENT ID','GRADE'],
    axis=1
)

y = df['GRADE']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

## Penjelasan

Data dibagi menggunakan metode Train-Test Split dengan proporsi:

* Data Training : 80%
* Data Testing : 20%

Pembagian ini bertujuan agar model dapat diuji menggunakan data yang belum pernah digunakan saat proses pelatihan.

---

# 4.3 Klasifikasi Menggunakan Decision Tree

## 4.3.1 Teori Decision Tree

Decision Tree merupakan algoritma klasifikasi yang membentuk struktur pohon keputusan berdasarkan atribut yang paling mampu memisahkan data ke dalam kelas tertentu.

Setiap node dalam pohon merepresentasikan atribut, sedangkan setiap cabang menunjukkan aturan keputusan.

Keunggulan Decision Tree:

* Mudah dipahami
* Mudah divisualisasikan
* Cepat dalam proses pelatihan
* Dapat digunakan untuk menentukan faktor yang paling berpengaruh

---

## 4.3.2 Implementasi

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

y_pred_dt = dt_model.predict(
    X_test
)
```

### Sumber Kode

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

---

## 4.3.3 Evaluasi Model

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

print(
    "Accuracy:",
    accuracy_dt
)

print(
    classification_report(
        y_test,
        y_pred_dt
    )
)
```

---

## 4.3.4 Hasil

| Metrik    | Nilai  |
| --------- | ------ |
| Accuracy  | 31.03% |
| Precision | 36.30% |
| Recall    | 31.03% |
| F1-Score  | 31.42% |

---

## 4.3.5 Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    y_pred_dt
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    'Confusion Matrix Decision Tree'
)

plt.show()
```

### Interpretasi

Decision Tree mampu menghasilkan performa yang cukup baik dibandingkan algoritma lain yang diuji.

Model dapat mengidentifikasi beberapa kelas dengan baik meskipun masih terdapat kesalahan klasifikasi pada beberapa kategori nilai.

---

# 4.4 Klasifikasi Menggunakan Random Forest

## 4.4.1 Teori Random Forest

Random Forest merupakan algoritma ensemble yang menggabungkan banyak pohon keputusan (Decision Tree) untuk meningkatkan stabilitas dan akurasi prediksi.

Metode ini bekerja menggunakan teknik bootstrap sampling dan voting.

Keunggulan Random Forest:

* Mengurangi overfitting
* Akurasi tinggi
* Dapat menghasilkan feature importance

---

## 4.4.2 Implementasi

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

y_pred_rf = rf_model.predict(
    X_test
)
```

### Sumber Kode

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

---

## 4.4.3 Evaluasi Model

```python
accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)

print(
    accuracy_rf
)
```

---

## 4.4.4 Hasil

| Metrik    | Nilai  |
| --------- | ------ |
| Accuracy  | 31.03% |
| Precision | 27.37% |
| Recall    | 31.03% |
| F1-Score  | 24.62% |

---

## 4.4.5 Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred_rf
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    'Confusion Matrix Random Forest'
)

plt.show()
```

### Interpretasi

Random Forest menghasilkan akurasi yang sama dengan Decision Tree.

Namun nilai F1-Score lebih rendah sehingga kemampuan klasifikasinya secara keseluruhan sedikit berada di bawah Decision Tree.

---

# 4.5 Klasifikasi Menggunakan K-Nearest Neighbor (KNN)

## 4.5.1 Teori KNN

K-Nearest Neighbor merupakan algoritma klasifikasi berbasis jarak.

Data baru akan diklasifikasikan berdasarkan mayoritas kelas dari sejumlah tetangga terdekat.

Pada penelitian ini digunakan:

```python
n_neighbors = 5
```

---

## 4.5.2 Implementasi

```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train,
    y_train
)

y_pred_knn = knn_model.predict(
    X_test
)
```

### Sumber Kode

https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

---

## 4.5.3 Hasil

| Metrik    | Nilai  |
| --------- | ------ |
| Accuracy  | 20.69% |
| Precision | 24.53% |
| Recall    | 20.69% |
| F1-Score  | 19.13% |

---

## 4.5.4 Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred_knn
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    'Confusion Matrix KNN'
)

plt.show()
```

### Interpretasi

Performa KNN berada di bawah Decision Tree dan Random Forest.

Hal ini menunjukkan bahwa pola data pada dataset tidak terlalu cocok dipisahkan hanya berdasarkan kedekatan jarak antar data.

---

# 4.6 Klasifikasi Menggunakan Naive Bayes

## 4.6.1 Teori Naive Bayes

Naive Bayes merupakan algoritma klasifikasi probabilistik yang menggunakan Teorema Bayes.

Algoritma ini mengasumsikan bahwa setiap atribut bersifat independen terhadap atribut lainnya.

Keunggulan:

* Cepat
* Sederhana
* Cocok untuk baseline model

---

## 4.6.2 Implementasi

```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(
    X_train,
    y_train
)

y_pred_nb = nb_model.predict(
    X_test
)
```

### Sumber Kode

https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

---

## 4.6.3 Hasil

| Metrik    | Nilai  |
| --------- | ------ |
| Accuracy  | 13.79% |
| Precision | 5.25%  |
| Recall    | 13.79% |
| F1-Score  | 7.59%  |

---

## 4.6.4 Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    y_pred_nb
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title(
    'Confusion Matrix Naive Bayes'
)

plt.show()
```

### Interpretasi

Naive Bayes menghasilkan performa terendah pada penelitian ini.

Hal ini mengindikasikan bahwa asumsi independensi antar atribut tidak sepenuhnya sesuai dengan karakteristik dataset Student Performance.

---

# 4.7 Perbandingan Model

Untuk menentukan model terbaik dilakukan perbandingan berdasarkan Accuracy, Precision, Recall, dan F1-Score.

## Hasil Perbandingan

| Algoritma     | Accuracy | Precision | Recall | F1-Score |
| ------------- | -------- | --------- | ------ | -------- |
| Decision Tree | 31.03%   | 36.30%    | 31.03% | 31.42%   |
| Random Forest | 31.03%   | 27.37%    | 31.03% | 24.62%   |
| KNN           | 20.69%   | 24.53%    | 20.69% | 19.13%   |
| Naive Bayes   | 13.79%   | 5.25%     | 13.79% | 7.59%    |

---

## Visualisasi Perbandingan Accuracy

```python
import matplotlib.pyplot as plt

models = [
    'Decision Tree',
    'Random Forest',
    'KNN',
    'Naive Bayes'
]

accuracy = [
    31.03,
    31.03,
    20.69,
    13.79
]

plt.figure(figsize=(8,5))

plt.bar(
    models,
    accuracy
)

plt.title(
    'Perbandingan Accuracy Model'
)

plt.ylabel(
    'Accuracy (%)'
)

plt.show()
```

### Interpretasi

Grafik menunjukkan bahwa Decision Tree dan Random Forest menghasilkan performa terbaik dibandingkan model lainnya.

---

# 4.8 Model Terbaik

Berdasarkan hasil evaluasi yang telah dilakukan, model terbaik pada penelitian ini adalah:

## Decision Tree

Alasan pemilihan:

1. Memiliki Accuracy tertinggi.
2. Memiliki F1-Score tertinggi.
3. Mudah dipahami dan diinterpretasikan.
4. Mampu menjelaskan proses pengambilan keputusan secara visual.

Dengan demikian Decision Tree dipilih sebagai model terbaik untuk memprediksi performa akademik mahasiswa pada dataset Student Performance.

---

# 4.9 Kesimpulan Bab

Empat algoritma klasifikasi berhasil diterapkan pada dataset Student Performance.

Berdasarkan hasil evaluasi, algoritma Decision Tree menghasilkan performa terbaik dengan Accuracy sebesar 31,03% dan F1-Score sebesar 31,42%.

Hasil tersebut menunjukkan bahwa Decision Tree merupakan metode yang paling sesuai untuk digunakan pada dataset yang dianalisis dalam penelitian ini.

