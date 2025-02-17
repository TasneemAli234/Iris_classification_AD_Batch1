from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)


model = joblib.load("gaussian_nb_model.pkl")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from the request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400
        
        # Convert the data into a DataFrame
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # أسماء الميزات كما تم تدريب النموذج عليها
        features = pd.DataFrame([data['features']], columns=feature_names)

        # Predict using the model
        prediction = model.predict(features)[0]

        # Convert the value to a regular float
        prediction = float(prediction)
        
        # Creat Dictionary to convert the number to name of class
        class_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_class = class_mapping.get(prediction, "Unknown")

        return jsonify({"predicted_class": predicted_class})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)