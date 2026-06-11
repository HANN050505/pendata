# pertemuan 15

# Explainability Model Peramalan NO₂ Menggunakan Skforecast dan LightGBM

## 1. Tujuan Analisis Prediksi

Analisis prediksi dilakukan untuk memperkirakan konsentrasi Nitrogen Dioksida (NO₂) pada periode waktu berikutnya berdasarkan pola historis yang terdapat pada data time series.

Model machine learning mempelajari hubungan antara nilai NO₂ pada periode sebelumnya dengan nilai NO₂ pada periode berikutnya sehingga dapat digunakan untuk melakukan forecasting kualitas udara di wilayah Bangkalan, Madura.

---

## 2. Bentuk Data Training

Pada awalnya data time series hanya terdiri dari satu kolom target.

### Data Awal

| Tanggal    | NO₂    |
| ---------- | ------ |
| 2025-01-01 | 220000 |
| 2025-01-02 | 225000 |
| 2025-01-03 | 230000 |
| 2025-01-04 | 228000 |
| 2025-01-05 | 235000 |

Agar dapat digunakan oleh algoritma machine learning, data diubah menjadi format supervised learning menggunakan teknik lag.

### Data Setelah Transformasi Lag

| lag_3  | lag_2  | lag_1  | Target |
| ------ | ------ | ------ | ------ |
| 220000 | 225000 | 230000 | 228000 |
| 225000 | 230000 | 228000 | 235000 |

Keterangan:

* lag_1 = nilai NO₂ satu periode sebelumnya
* lag_2 = nilai NO₂ dua periode sebelumnya
* lag_3 = nilai NO₂ tiga periode sebelumnya
* Target = nilai yang akan diprediksi

Input model (X):

```text
lag_1, lag_2, lag_3, ..., lag_n
```

Output model (y):

```text
NO₂ periode berikutnya
```

---

## 3. Apa Itu Lag?

Lag merupakan nilai historis dari suatu variabel yang digunakan sebagai informasi masa lalu untuk membantu model melakukan prediksi.

Contoh:

| Hari   | NO₂    |
| ------ | ------ |
| Senin  | 220000 |
| Selasa | 225000 |
| Rabu   | 230000 |
| Kamis  | 228000 |

Saat memprediksi hari Kamis:

| Fitur | Nilai  |
| ----- | ------ |
| lag_1 | 230000 |
| lag_2 | 225000 |
| lag_3 | 220000 |

Target:

```text
228000
```

Semakin relevan lag yang digunakan, semakin baik model dalam menangkap pola temporal.

---

## 4. Proses Analisis

### Tahap 1 - Membaca Dataset

```python
import pandas as pd

data = pd.read_csv("data_no2.csv")

print(data.head())
```

---

### Tahap 2 - Membagi Data

```python
train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]
```

---

### Tahap 3 - Membuat Model Forecasting

```python
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive

forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(
        random_state=123
    ),
    lags=24
)
```

Model menggunakan 24 lag terakhir sebagai fitur prediksi.

---

### Tahap 4 - Training Model

```python
forecaster.fit(y=train['NO2'])
```

Model mulai mempelajari pola historis dari data training.

---

### Tahap 5 - Melakukan Prediksi

```python
predictions = forecaster.predict(
    steps=len(test)
)

print(predictions.head())
```

---

### Tahap 6 - Evaluasi

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

mae = mean_absolute_error(
    test['NO2'],
    predictions
)

rmse = mean_squared_error(
    test['NO2'],
    predictions,
    squared=False
)

print("MAE :", mae)
print("RMSE:", rmse)
```

Interpretasi:

| Metrik | Fungsi                                                     |
| ------ | ---------------------------------------------------------- |
| MAE    | Rata-rata kesalahan absolut                                |
| RMSE   | Kesalahan dengan penalti lebih besar terhadap error tinggi |

Semakin kecil nilainya semakin baik.

---

# Explainability Model

Setelah model berhasil melakukan prediksi, langkah berikutnya adalah memahami alasan model menghasilkan prediksi tersebut.

---

## 5. Feature Importance

Feature Importance menunjukkan fitur mana yang paling berpengaruh terhadap prediksi.

### Kode

```python
importance = forecaster.get_feature_importances()

print(importance)
```

### Visualisasi

```python
import matplotlib.pyplot as plt

importance.sort_values(
    by='importance'
).plot(
    x='feature',
    y='importance',
    kind='barh',
    figsize=(10,6)
)

plt.title("Feature Importance")
plt.show()
```

### Contoh Hasil

| Feature | Importance |
| ------- | ---------- |
| lag_1   | 0.45       |
| lag_2   | 0.25       |
| lag_3   | 0.15       |
| lag_4   | 0.10       |
| lag_5   | 0.05       |

Interpretasi:

Nilai NO₂ pada satu periode sebelumnya (lag_1) memiliki pengaruh terbesar terhadap hasil prediksi.

---

## 6. SHAP Analysis

SHAP digunakan untuk menjelaskan kontribusi masing-masing fitur terhadap prediksi.

### Instalasi

```python
pip install shap
```

### Kode

```python
import shap

X_train, y_train = forecaster.create_train_X_y(
    y=train['NO2']
)

explainer = shap.TreeExplainer(
    forecaster.regressor
)

shap_values = explainer.shap_values(
    X_train
)
```

---

### SHAP Summary Plot

```python
shap.summary_plot(
    shap_values,
    X_train
)
```

Interpretasi:

* Titik merah menunjukkan nilai fitur tinggi.
* Titik biru menunjukkan nilai fitur rendah.
* Semakin jauh dari nol berarti semakin besar pengaruhnya terhadap prediksi.

---

## 7. Partial Dependence Plot (PDP)

PDP digunakan untuk melihat bagaimana perubahan suatu fitur mempengaruhi hasil prediksi.

---

### PDP untuk lag_1

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    figsize=(8,5)
)

PartialDependenceDisplay.from_estimator(
    estimator=forecaster.regressor,
    X=X_train,
    features=['lag_1'],
    kind='both',
    ax=ax
)

plt.show()
```

---

### PDP untuk Temperature dan lag_1

Jika dataset memiliki variabel cuaca:

```python
fig, ax = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12,5)
)

PartialDependenceDisplay.from_estimator(
    estimator=forecaster.regressor,
    X=X_train,
    features=['Temperature','lag_1'],
    kind='both',
    ax=ax
)

plt.tight_layout()
plt.show()
```

Interpretasi:

* Kurva naik → fitur meningkatkan prediksi.
* Kurva turun → fitur menurunkan prediksi.
* Garis putus-putus menunjukkan efek rata-rata.
* Garis tipis menunjukkan efek pada masing-masing observasi.

---

## 8. Ringkasan Hasil Analisis

| Analisis           | Tujuan                                    |
| ------------------ | ----------------------------------------- |
| Lag Analysis       | Mengetahui pengaruh data historis         |
| Feature Importance | Menentukan fitur paling penting           |
| SHAP               | Menjelaskan kontribusi tiap fitur         |
| PDP                | Mengetahui hubungan fitur dengan prediksi |

Hasil analisis menunjukkan bahwa model forecasting NO₂ sangat dipengaruhi oleh nilai historis pada periode sebelumnya, khususnya lag_1 dan lag_2. Feature Importance dan SHAP menunjukkan bahwa kedua fitur tersebut memiliki kontribusi terbesar dalam proses prediksi. Sementara itu, Partial Dependence Plot memperlihatkan bagaimana perubahan nilai lag mempengaruhi hasil forecasting secara keseluruhan.

---

## 9. Kesimpulan

Model Skforecast dengan algoritma LightGBM berhasil digunakan untuk melakukan prediksi konsentrasi NO₂ berdasarkan data historis. Transformasi lag memungkinkan data time series digunakan sebagai data supervised learning. Explainability menggunakan Feature Importance, SHAP, dan Partial Dependence Plot memberikan pemahaman yang lebih baik mengenai proses pengambilan keputusan model sehingga hasil prediksi tidak hanya akurat tetapi juga dapat dijelaskan secara ilmiah.
