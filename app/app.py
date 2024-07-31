import joblib
from flask import Flask, request, jsonify

# Load the best model
best_model = joblib.load('models/churn_model.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = best_model.predict([list(data.values())])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
