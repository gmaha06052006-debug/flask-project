from flask import Flask, render_template, request, jsonify
import joblib
import os
from preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = 'models/model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return True
    return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        if not load_models():
            return jsonify({'error': 'Model not trained yet!'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess
    cleaned = clean_text(text)
    
    # Vectorize
    vec = vectorizer.transform([cleaned])
    
    # Predict Label
    prediction = model.predict(vec)[0]
    
    # Get Confidence
    probs = model.predict_proba(vec)[0]
    # Classes are usually ['FAKE', 'REAL']
    confidence = float(max(probs)) * 100
    
    # Extract Key Indicators (Feature Importance)
    # Get local feature importance for this specific prediction
    feature_names = vectorizer.get_feature_names_out()
    # For Logistic Regression, coefficients represent importance
    # We find non-zero features in the input vector
    nonzero_indices = vec.nonzero()[1]
    
    # Get scores for these words
    scores = []
    for idx in nonzero_indices:
        # Score = abs(coefficient) * frequency_in_text
        score = model.coef_[0][idx]
        word = feature_names[idx]
        scores.append({'word': word, 'score': score})
    
    # Sort by influence (absolute value)
    scores.sort(key=lambda x: abs(x['score']), reverse=True)
    top_indicators = [s['word'] for s in scores[:5]]
    
    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 1),
        'indicators': top_indicators,
        'text': text[:100] + "..."
    })

if __name__ == '__main__':
    # Try to load models at startup
    load_models()
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
