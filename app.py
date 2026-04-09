from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model + scaler
try:
    with open("model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    print("✅ Model loaded")
except Exception as e:
    model, scaler = None, None
    print("❌ Error:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return render_template('index.html',
                                   prediction_text="❌ Model not loaded")

        # Get input values
        features = [float(x) for x in request.form.values()]

        if len(features) != 30:
            return render_template('index.html',
                                   prediction_text="❌ Enter all 30 values")

        # Convert & scale
        final_features = np.array(features).reshape(1, -1)
        final_features = scaler.transform(final_features)

        # Predict
        prediction = model.predict(final_features)

        # Result
        if prediction[0] == 1:
            result = '<span class="benign">✅ Benign (No Cancer)</span>'
        else:
            result = '<span class="malignant">⚠️ Malignant (Cancer Detected)</span>'

        return render_template('index.html', prediction_text=result)

    except ValueError:
        return render_template('index.html',
                               prediction_text="❌ Enter valid numbers")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"❌ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)