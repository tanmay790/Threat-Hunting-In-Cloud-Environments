import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
# Load the entire pipeline (preprocessing + classifier)
model = joblib.load('nids_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Expect a JSON payload with all features named as in training data
    input_json = request.get_json(force=True)
    # Convert to DataFrame (1 row)
    input_df = pd.DataFrame([input_json])
    # Get prediction (0 or 1)
    pred = model.predict(input_df)
    return jsonify({'prediction': int(pred[0])})

if __name__ == '__main__':
    # Use PORT env var if provided (Railway/Heroku style), else default 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

