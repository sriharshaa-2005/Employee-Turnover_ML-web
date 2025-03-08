from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app) 

with open('employee_turnover_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Extract features from the JSON data
        features = [
            float(data['satisfaction_level']),
            float(data['last_evaluation']),
            int(data['number_project']),
            int(data['average_montly_hours']),
            int(data['time_spent_company']),
            int(data['Work_accident']),
            int(data['promotion_last_5years']),
            int(data['department']),
            int(data['salary'])
        ]

        # Convert features to a NumPy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(features_array)

        # Return the prediction result as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
