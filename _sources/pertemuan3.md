# Pertemuan 3
## 1. Perbedaan Fitur dan Label

Sebelum melakukan pemodelan data (*Data Modeling*), kita harus memahami dua komponen utama dalam struktur dataset:

* **Fitur (*Feature* / Atribut / Variabel Independen):** Karakteristik atau properti dari data yang digunakan sebagai petunjuk atau masukan (*input*) untuk melakukan prediksi. Dalam tabel, ini biasanya adalah kolom-kolom prediktor. 
  * *Contoh:* `umur`, `jenis_kelamin`, `total_belanja`.
* **Label (Target / Kelas / Variabel Dependen):** Variabel yang ingin kita tebak atau prediksi (*output*). Label bergantung pada nilai-nilai fitur. 
  * *Contoh:* `loyal` (Yes/No).

---

## 2. Data Understanding Lanjutan

Kualitas data dunia nyata jarang sekali sempurna. Oleh karena itu, kita perlu menangani dua masalah utama di tahap awal:

### A. Missing Values (Data Kosong)
Terjadi ketika tidak ada nilai yang tersimpan untuk observasi tertentu dalam suatu kolom. Di Orange, data kosong ini ditandai dengan simbol tanda tanya (**?**).
* **Penyebab:** Kesalahan input, sensor rusak, atau responden menolak menjawab.
* **Penanganan di Orange:** Menggunakan widget **Impute**. Kita bisa mengisi data yang kosong dengan nilai rata-rata (*Mean*) untuk data numerik, nilai yang paling sering muncul (*Modus*) untuk data kategorikal, atau menghapus baris tersebut.

> **Visualisasi Data Kosong:**
> ![Screenshot Missing Values di Data Table](1.png)
> *Keterangan: Tampilan widget Data Table yang menunjukkan nilai '?' pada kolom umur/total belanja.*

### B. Outlier (Pencilan)
Data yang nilainya menyimpang sangat jauh dari mayoritas data lainnya (misalnya, nilai umur 250 tahun).
* **Penyebab:** Kesalahan pengukuran, *human error*, atau kejadian ekstrem yang nyata.
* **Penanganan di Orange:** Dideteksi secara visual menggunakan widget **Box Plot** atau **Distributions**. Titik-titik yang berada jauh di luar batas normal adalah *outlier*.

> **Visualisasi Outlier:**
> ![Screenshot Box Plot Outlier](2.png)
> *Keterangan: Tampilan widget Box Plot/Distributions yang menunjukkan adanya data ekstrem pada variabel umur.*

---

## 3. Menarik Data dari Berbagai Sumber (Database & JSON)

Sebagai pengolah data, kita harus mampu menarik data dari berbagai sumber:

* **Database Relasional (PostgreSQL):**
  Di Orange Data Mining, kita menggunakan widget **SQL Table**. Parameter koneksi yang digunakan adalah:
  * **Server / Host:** `localhost`
  * **Port:** `5432` 
  * **Database, Username, & Password:** Sesuai kredensial PostgreSQL.

> **Koneksi PostgreSQL:**
> ![Screenshot SQL Table](3.png)
> *Keterangan: Konfigurasi widget SQL Table yang berhasil terhubung ke database PostgreSQL.*

* **Format JSON:**
  Format pertukaran data standar dari API. Orange dapat membaca file `.json` secara langsung menggunakan widget **File**.

---

## 4. Mengukur Jarak pada Data Campuran

Algoritma dasar dalam Data Mining bekerja dengan mengukur kedekatan jarak antar baris data. 
* Data murni numerik: **Euclidean** / **Manhattan**.
* Data murni kategorikal: **Hamming** / **Jaccard**.

**Bagaimana jika datanya campuran (Numerik, Kategorikal, Biner, Ordinal)?**
Kita menggunakan **Gower Distance**. Metrik ini mengevaluasi setiap variabel sesuai tipe datanya (numerik dinormalisasi, kategorikal dicari kecocokan persisnya), lalu menggabungkannya menjadi satu nilai jarak berskala **0 hingga 1**.

### Contoh Dataset Campuran

| ID | Umur (Numerik) | Gender (Biner) | Pendidikan (Ordinal) | Produk (Kategorikal) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 25 | Pria | Sarjana | Elektronik |
| 2 | 45 | Wanita | SMA | Pakaian |
| 3 | 27 | Pria | Sarjana | Elektronik |
| 4 | 50 | Pria | SMP | Makanan |

---

## 5. Implementasi Pengukuran Jarak di Orange

Orange Data Mining menangani data campuran secara otomatis tanpa perlu menghitung rumus manual.

**Langkah-langkah Eksekusi:**
1. Tarik data dari PostgreSQL menggunakan **SQL Table**.
2. Hubungkan ke widget **Select Columns** untuk mendefinisikan *Features*, *Target* (Label), dan *Meta* (ID).
3. Hubungkan ke widget **Distances**.
4. Di dalam widget **Distances**, pilih metrik **Gower** untuk perhitungan data campuran.
5. Hubungkan ke widget **Distance Matrix** untuk melihat hasil perhitungan.
---

## 6. Interpretasi Hasil (Membaca Distance Matrix)

Setelah proses selesai, kita menganalisis tabel silang pada widget **Distance Matrix**.

> **Hasil Pengukuran Jarak:**
> ![Screenshot Distance Matrix](4.png)
> *Keterangan: Matriks jarak antar baris data (pelanggan).*

**Cara Membacanya:**
* **Mendekati 0:** Jika jarak antara Pelanggan A dan B sangat kecil (misal: 0.05), profil keduanya **sangat mirip**.
* **Mendekati 1:** Jika jaraknya mendekati 1 (misal: 0.95), kedua profil tersebut **sangat berbeda**.

## 7. Kesimpulan
Sebelum data diproses menggunakan algoritma machine learning, tahap *Data Understanding* dan *Preprocessing* sangatlah krusial. Memisahkan *Fitur* dan *Label*, menangani *Missing Values*, mengidentifikasi *Outlier*, serta memahami tipe data (numerik, ordinal, nominal, biner) adalah syarat wajib agar pengukuran jarak (*Gower Distance*) menghasilkan analisis yang valid dan akurat.