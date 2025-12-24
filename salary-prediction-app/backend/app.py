from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# load model & encoders
model = joblib.load(os.path.join(BASE_DIR, "backend", "salary_model_web.joblib"))
job_freq = joblib.load(os.path.join(BASE_DIR, "backend", "job_title_freq.joblib"))
location_cols = joblib.load(os.path.join(BASE_DIR, "backend", "location_columns.joblib"))

@app.route("/")
def home():
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    job_title_encoded = job_freq.get(data["job_title"], 0)

    loc_vector = [0]*len(location_cols)
    loc_name = f"company_location_top15_{data['company_location']}"
    if loc_name in location_cols:
        loc_vector[location_cols.index(loc_name)] = 1

    features = [
        data["work_year"],
        job_title_encoded,
        data["remote_ratio"],
        data["experience_level"],
        data["company_size"],
        data["employee_residence_freq"]
    ] + loc_vector

    prediction = model.predict([features])[0]

    return jsonify({
        "prediction": round(float(prediction), 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
