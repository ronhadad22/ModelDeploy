import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model/model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    prediction = model.predict([features])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
