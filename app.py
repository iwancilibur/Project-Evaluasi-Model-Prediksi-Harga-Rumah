# Import library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv('data/Housing.csv')
print(df.head())
print(df.info())


#Analisis Data dan Visualisasi
#=================================================================================#
# Eksplorasi data awal
print(df.describe())

# Visualisasi distribusi harga rumah (target variable)
plt.figure(figsize=(10,6))
sns.histplot(df['price'], kde=True)
plt.title('Distribusi Harga Rumah')
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
plt.show()

# Korelasi antara fitur numerik
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
plt.figure(figsize=(10,8))
sns.heatmap(df[numerical_features + ['price']].corr(), annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi Fitur Numerik')
plt.show()

# Analisis fitur kategorikal
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

plt.figure(figsize=(15,10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3,3,i)
    sns.boxplot(x=feature, y='price', data=df)
    plt.title(f'Harga berdasarkan {feature}')
plt.tight_layout()
plt.show()

#Preprocessing Data
#=================================================================================#
# Pisahkan fitur dan target
X = df.drop('price', axis=1)
y = df['price']

# Bagi data menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Gabungkan preprocessing dengan model dalam pipeline
models = {
    'Linear Regression': Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(random_state=42))])
}

#Pelatihan dan Evaluasi Model
#=================================================================================#
results = {}

for name, model in models.items():
    # Training model
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std()
    }
    
    # Visualisasi prediksi vs aktual
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title(f'Prediksi vs Aktual - {name}')
    plt.show()

# Tampilkan hasil evaluasi
results_df = pd.DataFrame(results).T
print(results_df)

#Kesimpulan Evaluasi Model
#=================================================================================#
# Analisis hasil evaluasi
print("\nKesimpulan Evaluasi Model:")

# Bandingkan performa model
best_model = results_df['R2'].idxmax()
print(f"\nModel terbaik berdasarkan R-squared adalah: {best_model}")

# Analisis metrik
print("\nAnalisis metrik evaluasi:")
print("1. MAE (Mean Absolute Error): Mengukur rata-rata kesalahan absolut prediksi.")
print("2. RMSE (Root Mean Squared Error): Memberikan bobot lebih besar untuk kesalahan besar.")
print("3. R-squared: Mengukur proporsi variasi dalam data yang dijelaskan oleh model.")
print("4. Cross-validation: Memastikan model tidak overfitting.")

# Kelebihan dan kekurangan model
print("\nKelebihan dan Kekurangan Model:")
print("- Linear Regression:")
print("  + Kelebihan: Mudah diinterpretasi, cepat")
print("  - Kekurangan: Tidak menangani hubungan non-linear dengan baik")
print("- Random Forest:")
print("  + Kelebihan: Bisa menangani hubungan non-linear, robust terhadap outliers")
print("  - Kekurangan: Lebih kompleks, butuh tuning hyperparameter")

# Rekomendasi
print("\nRekomendasi:")
print(f"Berdasarkan hasil evaluasi, model {best_model} menunjukkan performa terbaik.")
print("Untuk peningkatan lebih lanjut:")
print("1. Lakukan feature engineering untuk mengekstrak informasi lebih dari data")
print("2. Tuning hyperparameter untuk model Random Forest")
print("3. Coba model lain seperti Gradient Boosting atau Neural Networks")
print("4. Kumpulkan lebih banyak data jika memungkinkan")

#Penyimpanan Hasil
#=================================================================================#
# Simpan hasil evaluasi ke file
results_df.to_csv('hasil_evaluasi_model.csv')

# Simpan visualisasi
plt.figure(figsize=(10,6))
results_df[['R2', 'CV R2 Mean']].plot(kind='bar')
plt.title('Perbandingan R-squared Model')
plt.ylabel('Nilai R-squared')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('perbandingan_r2.png')
plt.show()