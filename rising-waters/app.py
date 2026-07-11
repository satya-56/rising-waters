from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/flood_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["Temp"]),
        float(request.form["Humidity"]),
        float(request.form["CloudCover"]),
        float(request.form["Annual"]),
        float(request.form["JanFeb"]),
        float(request.form["MarMay"]),
        float(request.form["JunSep"]),
        float(request.form["OctDec"]),
        float(request.form["AvgJune"]),
        float(request.form["Sub"])
    ]

    prediction = model.predict(pd.DataFrame([values]))[0]

    if prediction == 1:
        result = "Flood Likely"
    else:
        result = "No Flood"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)