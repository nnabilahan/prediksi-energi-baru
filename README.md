# 🔋 Aplikasi Prediksi Energi Terbarukan

Aplikasi ini menggunakan Flask dan machine learning untuk memprediksi potensi energi terbarukan berdasarkan data historis. Visualisasi hasil prediksi ditampilkan dalam bentuk grafik interaktif di halaman web.

---

## 🛠️ Langkah Instalasi

### 1. Clone Repositori

```bash gunakan terminal VSCode
git clone https://github.com/mkeyzxi/prediksi-potensi-energi-terbarukan.git
cd prediksi-potensi-energi-terbarukan
pip install flask pandas numpy prophet scikit-learn matplotlib
source .venv/bin/activate
python app.py

```

### 📁 Struktur Proyek
```
prediksi-potensi-energi-terbarukan/
│
├── app.py                # File utama aplikasi Flask
├── templates/            # Folder untuk file HTML
│   └── index.html        # Halaman utama aplikasi
│
├── static/               # Folder untuk file statis (CSS, JS)
│   └── css/
│       └── style.css     # File CSS untuk styling
│
└── data/                 # Folder untuk menyimpan data historis
    └── data.csv          # Contoh file data historis
```
### 2. Akses Aplikasi
Buka browser dan akses aplikasi di alamat berikut:

```Run
http://127.0.0.1:5000
```
