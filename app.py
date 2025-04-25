from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import numpy as np
from prophet import Prophet
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ==================== DATA PREPROCESSING ====================
df = pd.read_csv('data/Prediksi_Potensi.csv')

# 🧹 Bersihkan dan normalisasi data
df["Sumber"] = df["Sumber"].str.strip().str.title()
df = df.drop_duplicates(subset=["Sumber", "Tahun"], keep="first")

# 🔁 Augmentasi data simulasi pertumbuhan 6 tahun ke depan
df_aug = []
np.random.seed(42)
for _, row in df.iterrows():
    sumber = row["Sumber"]
    potensi_awal = row["Potensi"]
    tahun_awal = row["Tahun"]
    for i in range(6):
        tahun = tahun_awal + i
        if sumber in ["Surya", "Angin"]:
            potensi = potensi_awal + i * np.random.uniform(1.5, 3.0)
        elif sumber in ["Bioenergi", "Mikrohidro"]:
            potensi = potensi_awal + i * np.random.uniform(0.8, 1.5)
        else:
            potensi = potensi_awal + i * np.random.uniform(-1.0, 1.0)
        df_aug.append([sumber, tahun, potensi])

df_aug = pd.DataFrame(df_aug, columns=["Sumber", "Tahun", "Potensi"])
df_aug["Sumber"] = df_aug["Sumber"].str.title()

# ==================== ROUTES ====================

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediksi')
def prediksi():
    sumber_list = sorted(df_aug["Sumber"].unique())
    return render_template("index.html", sumber_list=sumber_list)

@app.route('/grafik')
def grafik():
    return render_template("grafik.html")

@app.route('/predict', methods=['POST'])
def predict():
    sumber = request.form["sumber"]
    df_filtered = df_aug[df_aug["Sumber"] == sumber.title()].copy()

    df_prophet = df_filtered.rename(columns={"Tahun": "ds", "Potensi": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=6, freq="Y")
    forecast = model.predict(future)
    forecast_filtered = forecast[["ds", "yhat"]].tail(6)
    forecast_filtered["Tahun"] = forecast_filtered["ds"].dt.year

    result = {
        "sumber": sumber,
        "prediksi": forecast_filtered[["Tahun", "yhat"]].to_dict(orient="records")
    }
    return jsonify(result)

@app.route('/insight/<sumber>')
def insight(sumber):
    sumber_title = sumber.title()
    df_filtered = df_aug[df_aug["Sumber"] == sumber_title].copy()

    df_prophet = df_filtered.rename(columns={"Tahun": "ds", "Potensi": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=6, freq="Y")
    forecast = model.predict(future)

    forecast_filtered = forecast[["ds", "yhat"]].tail(6)
    forecast_filtered["Tahun"] = forecast_filtered["ds"].dt.year

    delta = forecast_filtered["yhat"].diff().mean()
    tren = "meningkat" if delta > 0 else "menurun" if delta < 0 else "stagnan"
    potensi_akhir = forecast_filtered["yhat"].iloc[-1]
    tahun_akhir = forecast_filtered["Tahun"].iloc[-1]

    insight_text = f"""
    Potensi energi dari sumber {sumber_title} diperkirakan akan {tren} secara konsisten selama periode 2025 hingga {tahun_akhir}.
    Berdasarkan hasil prediksi Prophet, potensi tertinggi mencapai sekitar {potensi_akhir:.2f} MW pada tahun {tahun_akhir}.
    """
    return jsonify({"insight": insight_text.strip()})

@app.route('/download/<sumber>')
def download_pdf(sumber):
    sumber_title = sumber.title()
    df_filtered = df_aug[df_aug["Sumber"] == sumber_title].copy()

    df_prophet = df_filtered.rename(columns={"Tahun": "ds", "Potensi": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=6, freq="Y")
    forecast = model.predict(future)
    forecast_filtered = forecast[["ds", "yhat"]].tail(6)
    forecast_filtered["Tahun"] = forecast_filtered["ds"].dt.year

    delta = forecast_filtered["yhat"].diff().mean()
    tren = "meningkat" if delta > 0 else "menurun" if delta < 0 else "stagnan"
    potensi_akhir = forecast_filtered["yhat"].iloc[-1]
    tahun_akhir = forecast_filtered["Tahun"].iloc[-1]

    # PDF Export
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 770, f"Laporan Potensi Energi - {sumber_title}")
    p.setFont("Helvetica", 12)
    p.drawString(50, 740, "Insight:")
    p.drawString(65, 725, f"Potensi energi dari {sumber_title} diperkirakan akan {tren}.")
    p.drawString(65, 710, f"Prediksi mencapai {potensi_akhir:.2f} MW pada tahun {tahun_akhir}.")
    p.setFont("Helvetica-Bold", 12)
    p.drawString(60, 680, "Tahun")
    p.drawString(160, 680, "Potensi (MW)")

    y_pos = 660
    p.setFont("Helvetica", 12)
    for _, row in forecast_filtered.iterrows():
        p.drawString(60, y_pos, str(int(row["Tahun"])))
        p.drawString(160, y_pos, f"{row['yhat']:.2f}")
        y_pos -= 20

    p.showPage()
    p.save()
    buffer.seek(0)

    response = make_response(buffer.read())
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=laporan_{sumber}.pdf"
    return response

@app.route('/all_data')
def all_data():
    grouped_data = {}
    df_grouped = df_aug.groupby(["Sumber", "Tahun"], as_index=False)["Potensi"].mean()

    for sumber in df_grouped["Sumber"].unique():
        df_hist = df_grouped[df_grouped["Sumber"] == sumber][["Tahun", "Potensi"]].copy()

        df_filtered = df_aug[df_aug["Sumber"] == sumber].copy()
        df_prophet = df_filtered.rename(columns={"Tahun": "ds", "Potensi": "y"})
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=6, freq="Y")
        forecast = model.predict(future)
        forecast_filtered = forecast[["ds", "yhat"]].tail(6)
        forecast_filtered["Tahun"] = forecast_filtered["ds"].dt.year
        forecast_filtered.rename(columns={"yhat": "Potensi"}, inplace=True)

        df_combined = pd.concat([df_hist, forecast_filtered[["Tahun", "Potensi"]]])
        df_combined.sort_values("Tahun", inplace=True)

        grouped_data[sumber] = df_combined.to_dict(orient="records")

    return jsonify(grouped_data)

# ==================== RUN APP ====================
if __name__ == "__main__":
    app.run(debug=True)
