from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('best_model.pkl')  # Load your trained model

@app.route('/')
def home():
    return render_template('smartbridge.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = [float(x) for x in request.form.values()]
        prediction = model.predict([input_values])[0]

        if prediction == 1:
            result_text = "⚠️ Heart Disease Detected"
            result_class = "danger"
        else:
            result_text = "✅ No Heart Disease Detected"
            result_class = "success"

        return render_template(
            'smartbridge.html',
            prediction_text=result_text,
            result_class=result_class
        )
    except Exception as e:
        return render_template(
            'smartbridge.html',
            prediction_text=f"Error: {str(e)}",
            result_class="danger"
        )

if __name__ == '__main__':
    app.run(debug=True)
