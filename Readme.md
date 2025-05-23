# Project Evaluasi Model Prediksi Harga Rumah

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Proyek ini bertujuan untuk mengevaluasi model prediksi harga rumah menggunakan berbagai algoritma machine learning. Dataset yang digunakan adalah data properti dari India dengan 545 sampel.

## 📋 Daftar Isi
- [Dataset](#-dataset)
- [Analisis Data](#-analisis-data)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Evaluasi Model](#-evaluasi-model)
- [Kesimpulan](#-kesimpulan)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Lisensi](#-lisensi)

## 📊 Dataset

**Sumber Dataset**: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) dari Kaggle

### Karakteristik Dataset
- Jumlah data: 545 entri
- Tipe fitur:
  - **Numerik**: 
    - `area` (luas bangunan)
    - `bedrooms` (jumlah kamar tidur)
    - `bathrooms` (jumlah kamar mandi)
    - `stories` (jumlah lantai)
    - `parking` (jumlah tempat parkir)
  
  - **Kategorikal**:
    - `mainroad` (akses jalan utama)
    - `guestroom` (kamar tamu)
    - `basement` (basement)
    - `hotwaterheating` (pemanas air)
    - `airconditioning` (AC)
    - `prefarea` (daerah preferensi)
    - `furnishingstatus` (status furnishing)

- **Target**: `price` (harga rumah dalam satuan lakh)

### Contoh Data
| area | bedrooms | bathrooms | stories | parking | price | furnishingstatus |
|------|----------|-----------|---------|---------|-------|------------------|
| 7420 | 4        | 2         | 3       | 2       | 6.07  | furnished        |
| 8960 | 4        | 4         | 4       | 3       | 7.80  | furnished        |

## 📈 Analisis Data

### 1. Distribusi Harga Rumah
![Distribusi Harga](https://github.com/user-attachments/assets/543ff310-f968-4ca6-80e5-e58ff58a0b43)

Analisis menunjukkan distribusi harga rumah cenderung right-skewed, menunjukkan sebagian besar rumah memiliki harga di kisaran menengah.

### 2. Matriks Korelasi
![Matriks Korelasi](https://github.com/user-attachments/assets/b78049f3-7af9-4856-a872-3702789f7568)

Fitur `area` menunjukkan korelasi positif terkuat dengan harga (0.54), diikuti oleh `bathrooms` (0.52) dan `stories` (0.42).

### 3. Analisis Fitur Kategorikal
![Fitur Kategorikal](https://github.com/user-attachments/assets/94da6ae9-754b-4fc2-94c6-102f891bdde8)

Rumah dengan:
- Akses jalan utama (`mainroad`)
- Fasilitas AC (`airconditioning`)
- Daerah preferensi (`prefarea`)

cenderung memiliki harga lebih tinggi.

## 🤖 Model yang Digunakan

### 1. Linear Regression
- **Keunggulan**:
  - Sederhana dan mudah diinterpretasi
  - Waktu pelatihan cepat
  - Performa bagus untuk hubungan linear
  
- **Keterbatasan**:
  - Tidak bisa menangkap hubungan non-linear
  - Sensitif terhadap outliers

### 2. Random Forest Regressor
- **Keunggulan**:
  - Bisa menangkap hubungan non-linear
  - Robust terhadap outliers
  - Tidak memerlukan feature scaling
  
- **Keterbatasan**:
  - Lebih kompleks
  - Cenderung overfitting jika tidak di-tuning dengan baik
  - Waktu prediksi lebih lama

## 📊 Evaluasi Model

### Metrik Evaluasi
| Model              | MAE    | RMSE   | R²     | CV R² (Mean ± Std) |
|--------------------|--------|--------|--------|--------------------|
| Linear Regression  | 9.70   | 1.32   | 0.65   | 0.64 ± 0.03        |
| Random Forest      | 1.02   | 1.40   | 0.61   | 0.60 ± 0.03        |

### Visualisasi Hasil
**Linear Regression**:
![Linear Regression](https://github.com/user-attachments/assets/f1df7434-a324-48d0-99a6-810b7755c161)

**Random Forest**:
![Random Forest](https://github.com/user-attachments/assets/34d66469-c31f-4831-9c60-e161b003a182)

## 🎯 Kesimpulan

### Temuan Utama
1. **Linear Regression** menunjukkan performa lebih baik dengan:
   - Nilai R² lebih tinggi (0.65 vs 0.61)
   - RMSE lebih rendah (1.32 vs 1.40)
   - Konsistensi lebih baik dalam cross-validation

2. **Fitur Paling Berpengaruh**:
   - Luas bangunan (`area`)
   - Jumlah kamar mandi (`bathrooms`)
   - Fasilitas AC (`airconditioning`)

### Rekomendasi Pengembangan
1. **Feature Engineering**:
   - Coba transformasi logaritmik untuk fitur `area` dan `price`
   - Buat fitur interaksi antara `area` dan `bathrooms`
   
2. **Model Improvement**:
   - Tuning hyperparameter Random Forest
   - Coba model Gradient Boosting (XGBoost, LightGBM)
   - Evaluasi model Neural Networks sederhana

3. **Data Collection**:
   - Tambahkan lebih banyak sampel
   - Pertimbangkan fitur lokasi/geografis

## 💻 Instalasi

### Prasyarat
- Python 3.8+
- pip

### Langkah-langkah
1. Clone repository:
```bash
git clone https://github.com/iwancilibur/Project-Evaluasi-Model-Prediksi-Harga-Rumah.git
cd Project-Evaluasi-Model-Prediksi-Harga-Rumah
```
