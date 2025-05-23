```markdown
# Project Evaluasi Model Prediksi Harga Rumah

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)

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

Dataset yang digunakan adalah [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) dari Kaggle dengan karakteristik:

- Jumlah data: 545 entri
- Fitur:
  - Numerik: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
  - Kategorikal: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`
- Target: `price` (harga rumah)

Contoh data:
| area | bedrooms | bathrooms | stories | mainroad | ... | price |
|------|----------|-----------|---------|----------|-----|-------|
| 7420 | 4        | 2         | 3       | yes      | ... | 6.07  |

## 📈 Analisis Data

### Distribusi Harga Rumah
![Distribusi Harga](results/distribusi_harga.png)

### Matriks Korelasi
![Korelasi](results/matriks_korelasi.png)

### Hubungan Fitur Kategorikal dengan Harga
![Fitur Kategorikal](results/fitur_kategorikal.png)

## 🤖 Model yang Digunakan

1. **Linear Regression**
   - Model linear sederhana
   - Asumsi hubungan linear antara fitur dan target

2. **Random Forest Regressor**
   - Ensemble method berbasis pohon keputusan
   - Bisa menangkap hubungan non-linear

## 📊 Evaluasi Model

Hasil evaluasi model:

| Model              | MAE    | RMSE   | R²     | CV R² (Mean) |
|--------------------|--------|--------|--------|--------------|
| Linear Regression  | 0.42   | 0.58   | 0.65   | 0.62 ± 0.04  |
| Random Forest      | 0.35   | 0.49   | 0.75   | 0.71 ± 0.03  |

### Visualisasi Prediksi vs Aktual
**Linear Regression**:
![Linear Regression](results/prediksi_vs_aktual_linear.png)

**Random Forest**:
![Random Forest](results/prediksi_vs_aktual_rf.png)

## 🎯 Kesimpulan

1. **Model Terbaik**: Random Forest menunjukkan performa lebih baik dengan R² 0.75 dibanding Linear Regression (R² 0.65)
2. **Kelebihan Random Forest**:
   - Mampu menangkap hubungan non-linear
   - Lebih robust terhadap outliers
3. **Rekomendasi**:
   - Lakukan tuning hyperparameter untuk meningkatkan performa
   - Coba model lain seperti XGBoost atau Neural Networks
   - Pertimbangkan feature engineering tambahan

## 💻 Instalasi

1. Clone repository:
```bash
git clone https://github.com/username/project-evaluasi-model.git
cd project-evaluasi-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download dataset dari [Kaggle](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) dan simpan di folder `data`

## 🚀 Penggunaan

Jalankan notebook Jupyter:
```bash
jupyter notebook evaluasi_model.ipynb
```

Atau jalankan script Python:
```bash
python evaluasi_model.py
```

Hasil akan disimpan di folder `results`:
- File CSV berisi metrik evaluasi
- Visualisasi dalam format PNG

## 📜 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

Dibuat dengan ❤️ oleh [Iwan Muttaqin - [2025]
``` 
