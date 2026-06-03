# Peramalan Kadar Nitrogen Dioksida (NO₂) di Kabupaten Bangkalan Menggunakan Algoritma K-Nearest Neighbor (KNN)

## Latar Belakang

Pencemaran udara merupakan salah satu permasalahan lingkungan yang terus meningkat seiring bertambahnya aktivitas manusia. Pertumbuhan sektor transportasi, industri, pembangkit energi, dan kepadatan penduduk menyebabkan peningkatan emisi berbagai polutan ke atmosfer. Salah satu polutan yang banyak mendapat perhatian adalah Nitrogen Dioksida (NO₂).

NO₂ merupakan gas berbahaya yang dihasilkan dari proses pembakaran bahan bakar fosil, seperti kendaraan bermotor, aktivitas industri, dan pembangkit listrik. Paparan NO₂ dalam jangka panjang dapat menyebabkan gangguan sistem pernapasan, memperparah penyakit asma, menurunkan fungsi paru-paru, serta berkontribusi terhadap pembentukan hujan asam dan smog fotokimia.

Kabupaten Bangkalan merupakan salah satu wilayah strategis di Pulau Madura yang mengalami perkembangan aktivitas ekonomi dan transportasi yang cukup pesat. Kondisi tersebut berpotensi memengaruhi kualitas udara, khususnya kadar NO₂ di atmosfer. Oleh karena itu, diperlukan suatu metode yang mampu memprediksi kadar NO₂ di masa mendatang sehingga dapat digunakan sebagai dasar pengambilan keputusan dalam pengelolaan kualitas udara.

Pada penelitian ini dilakukan proses peramalan kadar NO₂ harian di Kabupaten Bangkalan menggunakan data satelit Sentinel-5P yang diperoleh melalui platform OpenEO Copernicus Data Space Ecosystem. Data kemudian diproses melalui tahapan preprocessing, interpolasi missing value, deteksi outlier menggunakan metode Interquartile Range (IQR), transformasi data time series menjadi supervised learning, hingga pembangunan model menggunakan algoritma K-Nearest Neighbor (KNN). Selain melakukan prediksi satu hari ke depan, penelitian ini juga mengembangkan prediksi kadar NO₂ hingga tiga hari mendatang menggunakan pendekatan recursive forecasting.

---

# Pengambilan Data NO₂

Data NO₂ diperoleh dari satelit Sentinel-5P melalui layanan OpenEO Copernicus Data Space Ecosystem.

### Instalasi Library

```python
pip install openeo
```

### Koneksi ke OpenEO

```python
import openeo

connection = openeo.connect(
    "openeo.dataspace.copernicus.eu"
).authenticate_oidc()
```

Saat kode dijalankan akan muncul proses autentikasi.

```text
Visit https://identity.dataspace.copernicus.eu/auth/...
to authenticate

✅ Authorized successfully
Authenticated using device code flow.
```

---

## Menentukan Area Kabupaten Bangkalan

```python
aoi = {
    "type": "Polygon",
    "coordinates": [
        [
            [113.09, -6.89],
            [112.68, -6.89],
            [112.68, -7.20],
            [113.09, -7.20],
            [113.09, -6.89]
        ]
    ]
}
```

---

## Mengambil Data Sentinel-5P

```python
s5post = connection.load_collection(
    "SENTINEL_5P_L2",
    temporal_extent=[
        "2024-10-01",
        "2026-06-03"
    ],
    spatial_extent={
        "west":112.68,
        "south":-7.20,
        "east":113.09,
        "north":-6.89
    },
    bands=["NO2"]
)
```

---

## Agregasi Temporal Harian

```python
s5p_no2_daily = s5post.aggregate_temporal_period(
    reducer="mean",
    period="day"
)
```

---

## Agregasi Spasial Area Bangkalan

```python
s5p_no2_aoi = s5p_no2_daily.aggregate_spatial(
    reducer="mean",
    geometries=aoi
)
```

---

## Menjalankan Batch Job

```python
job = s5post.execute_batch(
    title="NO2 Bangkalan Terbaru",
    outputfile="NO2Bangkalan_Terkini.nc"
)
```

File hasil pengambilan data akan tersimpan dalam format NetCDF (`.nc`).

---

# Preprocessing Data

Tahap preprocessing dilakukan untuk mengubah data mentah satelit menjadi data time series yang siap digunakan pada proses pemodelan.

---

## Membaca File NetCDF

```python
import netCDF4

file_path = "NO2Bangkalan_Terkini.nc"

ds = netCDF4.Dataset(file_path)

print(ds.variables.keys())
```

Output:

```text
dict_keys(['t','x','y','crs','NO2'])
```

---

## Mengambil Variabel NO₂ dan Waktu

```python
no2 = ds.variables["NO2"][:]

time = ds.variables["t"][:]

time_units = ds.variables["t"].units

dates = netCDF4.num2date(
    time,
    units=time_units
)
```

---

## Mengatasi Missing Value dengan Interpolasi Linear

```python
import numpy as np
import pandas as pd

no2_filled = np.zeros_like(no2)

for i in range(no2.shape[1]):
    for j in range(no2.shape[2]):

        series = pd.Series(
            no2[:,i,j]
        )

        no2_filled[:,i,j] = (
            series.interpolate(
                method="linear",
                limit_direction="both"
            ).to_numpy()
        )
```

---

## Mengubah Menjadi Data Harian

```python
new_dates = []
new_no2 = []

for i in range(len(dates)):

    new_dates.append(
        dates[i].strftime("%Y-%m-%d")
    )

    new_no2.append(
        np.mean(no2_filled[i])
    )
```

---

## Menyimpan Menjadi CSV

```python
df = pd.DataFrame({
    "date": new_dates,
    "NO2": new_no2
})

df.to_csv(
    "NO2_Bangkalan.csv",
    index=False
)
```

---

# Pemeriksaan Missing Value Time Series

```python
df = pd.read_csv(
    "NO2_Bangkalan.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

full_range = pd.date_range(
    start="2024-10-01",
    end="2026-06-03",
    freq="D"
)

missing_dates = (
    full_range.difference(
        df["date"]
    )
)

print(missing_dates)
```

---

## Mengisi Missing Date

```python
df = (
    df.set_index("date")
      .reindex(full_range)
)

df.index.name = "date"

df["NO2"] = (
    df["NO2"]
    .interpolate(method="time")
)

df["NO2"] = (
    df["NO2"]
    .bfill()
    .ffill()
)

df.to_csv(
    "NO2_Interpolated.csv"
)
```

---

# Deteksi dan Penanganan Outlier Menggunakan IQR

```python
Q1 = df["NO2"].quantile(0.25)

Q3 = df["NO2"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = (
    Q1 - 1.5 * IQR
)

upper_bound = (
    Q3 + 1.5 * IQR
)
```

---

## Mencari Outlier

```python
outliers = df[
    (df["NO2"] < lower_bound) |
    (df["NO2"] > upper_bound)
]

print(
    "Jumlah Outlier:",
    len(outliers)
)
```

---

## Menghapus dan Mengisi Kembali Outlier

```python
df["NO2_cleaned"] = (
    df["NO2"].mask(
        (df["NO2"] < lower_bound) |
        (df["NO2"] > upper_bound)
    )
)

df["NO2_filled"] = (
    df["NO2_cleaned"]
    .interpolate("linear")
)

df["NO2_filled"] = (
    df["NO2_filled"]
    .bfill()
)
```

---

# Transformasi Data Time Series

## Fungsi Membentuk Data Supervised

```python
import pandas as pd

def create_supervised(
    data,
    n_lag=4
):

    df_supervised = pd.DataFrame()

    for i in range(
        n_lag,
        0,
        -1
    ):
        df_supervised[
            f"NO2(t-{i})"
        ] = data.shift(i)

    df_supervised["NO2(t)"] = data

    df_supervised.dropna(
        inplace=True
    )

    return df_supervised
```

---

# Normalisasi Data

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["NO2_scaled"] = (
    scaler.fit_transform(
        df[["NO2_filled"]]
    )
)
```

---

# Analisis Korelasi Lag

```python
supervised_df30 = (
    create_supervised(
        df["NO2_scaled"],
        n_lag=30
    )
)

correlations = (
    supervised_df30
    .drop(columns="NO2(t)")
    .corrwith(
        supervised_df30["NO2(t)"]
    )
)

print(correlations)
```

Hasil menunjukkan bahwa lag 1 sampai lag 4 memiliki korelasi paling tinggi sehingga dipilih sebagai fitur utama dalam proses pemodelan.

---

# Pembentukan Dataset Model

## Dataset 4 Hari Sebelumnya

```python
supervised_df = create_supervised(
    df["NO2_scaled"],
    n_lag=4
)
```

---

## Dataset 10 Hari Sebelumnya

```python
supervised_df10 = create_supervised(
    df["NO2_scaled"],
    n_lag=10
)
```

---

# Pemodelan Menggunakan K-Nearest Neighbor (KNN)

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import numpy as np
```

---

## Fungsi MAPE

```python
def MAPE(
    y_true,
    y_pred
):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero = y_true != 0

    return np.mean(
        np.abs(
            (
                y_true[nonzero] -
                y_pred[nonzero]
            )
            /
            y_true[nonzero]
        )
    ) * 100
```

---

## Fungsi Training KNN

```python
def train_knn(
    df_supervised,
    model_name=""
):

    X = df_supervised.drop(
        columns=["NO2(t)"]
    ).values

    y = df_supervised[
        "NO2(t)"
    ].values

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False
        )
    )

    knn = KNeighborsRegressor(
        n_neighbors=5
    )

    knn.fit(
        X_train,
        y_train
    )

    y_pred = knn.predict(
        X_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = MAPE(
        y_test,
        y_pred
    )

    print(model_name)
    print("RMSE:", rmse)
    print("R2:", r2)
    print("MAPE:", mape)

    return knn
```

---

## Training Model

```python
knn_4 = train_knn(
    supervised_df,
    "KNN 4 Hari"
)

knn_10 = train_knn(
    supervised_df10,
    "KNN 10 Hari"
)
```

Berdasarkan hasil evaluasi, model dengan 4 hari sebelumnya menghasilkan performa terbaik dengan nilai RMSE lebih kecil, nilai R² lebih tinggi, dan nilai MAPE lebih rendah dibandingkan model 10 hari sebelumnya.

---

# Prediksi Kadar NO₂ Tiga Hari Ke Depan

Setelah model terbaik diperoleh, langkah berikutnya adalah melakukan prediksi kadar NO₂ selama tiga hari ke depan menggunakan metode Recursive Forecasting. Prediksi dilakukan dengan menggunakan hasil prediksi sebelumnya sebagai input untuk memprediksi hari berikutnya.

```python
last_4 = (
    df["NO2_scaled"]
    .tail(4)
    .values
)

future_predictions = []

current_window = list(last_4)

for i in range(3):

    pred = knn_4.predict(
        [current_window]
    )[0]

    future_predictions.append(
        pred
    )

    current_window.pop(0)

    current_window.append(pred)
```

---

## Mengembalikan Hasil ke Skala Asli

```python
future_predictions = (
    scaler.inverse_transform(
        np.array(
            future_predictions
        ).reshape(-1,1)
    )
)

print(
    "Prediksi 3 Hari Kedepan"
)

for i, pred in enumerate(
    future_predictions,
    start=1
):
    print(
        f"Hari +{i}: "
        f"{pred[0]:.8f}"
    )
```

Contoh output:

```text
Prediksi 3 Hari Kedepan

Hari +1 : 0.00005127
Hari +2 : 0.00005013
Hari +3 : 0.00004988
```

Nilai di atas merupakan estimasi kadar NO₂ harian untuk tiga hari mendatang berdasarkan pola historis yang dipelajari oleh model KNN.

---

# Kesimpulan

Penelitian ini berhasil membangun sistem peramalan kadar Nitrogen Dioksida (NO₂) di Kabupaten Bangkalan menggunakan data satelit Sentinel-5P dan algoritma K-Nearest Neighbor (KNN). Tahapan penelitian meliputi pengambilan data satelit, preprocessing, penanganan missing value menggunakan interpolasi linear, deteksi outlier menggunakan metode IQR, transformasi data time series menjadi supervised learning, normalisasi data, hingga proses pemodelan.

Berdasarkan hasil evaluasi, model KNN dengan empat hari sebelumnya sebagai fitur masukan menghasilkan performa terbaik dibandingkan model dengan sepuluh hari sebelumnya. Hal ini menunjukkan bahwa informasi dari beberapa hari terakhir memiliki pengaruh yang lebih kuat terhadap perubahan kadar NO₂ dibandingkan data yang lebih lama.

Model yang telah dibangun juga mampu digunakan untuk melakukan prediksi kadar NO₂ hingga tiga hari ke depan menggunakan pendekatan recursive forecasting. Hasil prediksi menunjukkan bahwa pola historis kadar NO₂ dapat dimanfaatkan untuk memperkirakan kondisi kualitas udara pada periode mendatang sehingga berpotensi mendukung proses monitoring dan pengambilan keputusan terkait pengelolaan lingkungan di Kabupaten Bangkalan.