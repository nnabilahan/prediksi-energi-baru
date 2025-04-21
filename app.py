from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Baca data dan siapkan preprocessing
df = pd.read_csv('data/Prediksi_Potensi.csv')

# Augmentasi data seperti sebelumnya
df_aug = []
np.random.seed(42)
for _, row in df.iterrows():
    sumber = row['Sumber']
    potensi_awal = row['Potensi']
    tahun_awal = row['Tahun']
    for i in range(6):
        tahun = tahun_awal + i
        if sumber in ['Surya', 'Angin']:
            potensi = potensi_awal + i * np.random.uniform(1.5, 3.0)
        elif sumber in ['Bioenergi', 'Mikrohidro']:
            potensi = potensi_awal + i * np.random.uniform(0.8, 1.5)
        else:
            potensi = potensi_awal + i * np.random.uniform(-1.0, 1.0)
        df_aug.append([sumber, tahun, potensi])

df_aug = pd.DataFrame(df_aug, columns=["Sumber", "Tahun", "Potensi"])
le = LabelEncoder()
df_aug["Sumber_encoded"] = le.fit_transform(df_aug["Sumber"])

@app.route('/')
def index():
    daftar_sumber = sorted(df_aug["Sumber"].unique())
    return render_template('index.html', sumber_list=daftar_sumber)

@app.route('/predict', methods=['POST'])
def predict():
    input_sumber = request.form['sumber'].strip().title()

    if input_sumber in df_aug["Sumber"].unique():
        df_sumber = df_aug[df_aug["Sumber"] == input_sumber][["Tahun", "Potensi"]]
        df_sumber = df_sumber.rename(columns={"Tahun": "ds", "Potensi": "y"})
        df_sumber["ds"] = pd.to_datetime(df_sumber["ds"], format="%Y")

        model = Prophet(yearly_seasonality=False, daily_seasonality=False)
        model.fit(df_sumber)

        future = model.make_future_dataframe(periods=6, freq='Y')
        forecast = model.predict(future)

        forecast["Tahun"] = forecast["ds"].dt.year
        forecast_filtered = forecast[(forecast["Tahun"] >= 2025) & (forecast["Tahun"] <= 2030)]

        result = {
            "sumber": input_sumber,
            "prediksi": forecast_filtered[["Tahun", "yhat"]].to_dict(orient='records')
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Sumber tidak ditemukan dalam data."}), 400

if __name__ == '__main__':
    app.run(debug=True)


# hapus aja ini untuk mbul -> membuat dan melatih model dengan format .pkl
# import pandas as pd
# import pickle
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Load model (misalnya, model prediksi yang sudah dilatih)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     try:
#         # Membaca data CSV
#         df = pd.read_csv('data/Prediksi_Potensi.csv')  # Sesuaikan dengan path file CSV Anda
        
#         # Prediksi dengan model (misalnya menggunakan model yang sudah dilatih)
#         forecast = model.predict(df)  # Gantilah dengan logika prediksi sesuai model Anda
#         forecast_filtered = forecast[['Tahun', 'yhat']]  # Mengambil kolom yang diperlukan

#         # Mengubah data menjadi format dictionary untuk dikirim ke template
#         chart_data = forecast_filtered.to_dict(orient='records')
        
#         return render_template('index.html', chart_data=chart_data)
    
#     except Exception as e:
#         return f"Error: {str(e)}"

# if __name__ == '__main__':
#     app.run(debug=True)
