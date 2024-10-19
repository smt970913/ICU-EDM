from flask import Flask, render_template_string, request, redirect, url_for, flash
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
import io
import os
from interactive_support import *

# Assume you have several trained models saved as files, here using torch as an example
# Please ensure you have the appropriate model files, such as 'model_1.pth', 'model_2.pth', ... 'model_7.pth'
MODEL_FILES = [
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_2_obj.pth',
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_3_obj.pth',
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_4_obj.pth',
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_5_obj.pth',
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_6_obj.pth',
    '../OCRL_fqe_model/offline_FQE_Agent_20241011_7_obj.pth'
]

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load the trained model
def load_model(model_index):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_FILES[model_index], map_location=device)
    return model

# HTML template for the index page
INDEX_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Select Model</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">Fitted-Q-Evaluation of Extubation Failure Rate (EFR)</h1>
    <div class="text-center mt-4">
        <img src="data:image/jpeg;base64,{{ img_data }}" alt="Illustration" class="img-fluid">
    </div>
    <form method="post" class="mt-4">
        <div class="form-group">
            <label for="model_selection">Select Model:</label>
            <select class="form-control" id="model_selection" name="model_selection">
                {% for i in range(1, model_files + 1) %}
                    <option value="{{ i }}">Model {{ i }}</option>
                {% endfor %}
            </select>
        </div>
        <button type="submit" class="btn btn-primary">Confirm Selection</button>
    </form>
</div>
</body>
</html>
"""

# HTML template for the predict page
PREDICT_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Enter Patient Data</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h1 class="text-center">Enter Patient Data</h1>
    <form method="post" class="mt-4">
        {% for i in range(labels|length) %}
        <div class="form-group">
            <label for="input_{{ i }}">{{ labels[i] }}</label>
            <input type="text" class="form-control" id="input_{{ i }}" name="input_{{ i }}" required>
        </div>
        {% endfor %}
        <button type="submit" class="btn btn-primary">Submit</button>
    </form>
    <div class="mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
            <div class="alert alert-{{ category }}" role="alert">
                {{ message }}
            </div>
            {% endfor %}
        {% endif %}
        {% endwith %}
    </div>
</div>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        let inputs = document.querySelectorAll("input");
        inputs.forEach((input, index) => {
            input.addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    event.preventDefault();
                    let nextInput = inputs[index + 1];
                    if (nextInput) {
                        nextInput.focus();
                    }
                }
            });
        });
    });
</script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_index = int(request.form.get('model_selection')) - 1
        return redirect(url_for('predict', model_index=model_index))

    # Load image and convert to base64 for HTML rendering
    img = Image.open('../image/illus_mechanical_vent.jpg')
    img = img.resize((400, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return render_template_string(INDEX_HTML_TEMPLATE, model_files=len(MODEL_FILES), img_data=img_str)

@app.route('/predict/<int:model_index>', methods=['GET', 'POST'])
def predict(model_index):
    model = load_model(model_index)
    scaler = StandardScaler()
    labels = [
        'Age (years)', 'Gender (Male=1, Female=0)', 'Weight (kg)', 'Heart Rate (bpm)',
        'Arterial O2 Pressure (mmHg)', 'Hemoglobin (g/dL)', 'Arterial CO2 Pressure (mmHg)',
        'Hematocrit (serum %)', 'White Blood Cell Count (WBC, x10^9/L)', 'Chloride (serum, mEq/L)',
        'Creatinine (serum, mg/dL)', 'Glucose (serum, mg/dL)', 'Magnesium (mg/dL)', 'Sodium (serum, mEq/L)',
        'Arterial pH', 'Inspired Oxygen Fraction (FiO2, %)', 'Arterial Base Excess (mmol/L)',
        'Blood Urea Nitrogen (BUN, mg/dL)', 'Potassium (serum, mEq/L)', 'Bicarbonate (HCO3, mEq/L)',
        'Platelet Count (x10^9/L)', 'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
        'Mean Blood Pressure (mmHg)', 'Temperature (°C)', 'Oxygen Saturation (SaO2, %)',
        'Glasgow Coma Scale (GCS) Score', 'Positive End-Expiratory Pressure (PEEP, cmH2O)',
        'Respiratory Rate (breaths/min)', 'Tidal Volume (mL)'
    ]

    if request.method == 'POST':
        try:
            input_values = [float(request.form.get(f"input_{i}")) for i in range(len(labels))]
            if len(input_values) != 30:
                flash("Please enter all 30 variables, including the patient's gender.", "error")
                return redirect(url_for('predict', model_index=model_index))

            # Split input into features and gender
            gender = input_values[1]
            features = np.array([input_values[0]] + input_values[2:]).reshape(1, -1)

            # Standardize the features
            scaled_features = scaler.fit_transform(features)

            # Add gender information to the standardized features at the correct position
            final_input = np.insert(scaled_features, 1, gender, axis=1)

            # Convert input to torch tensor
            final_input_tensor = torch.tensor(final_input, dtype=torch.float).to(device)
            print("Input tensor shape:", final_input_tensor.shape)

            # Use the model to make predictions and return the result
            response = model.avg_Q_value_est(final_input_tensor)
            flash(f"Model Response: {response}", "success")
        except ValueError:
            flash("Invalid input, please ensure all values are numeric.", "error")

    return render_template_string(PREDICT_HTML_TEMPLATE, labels=labels)

if __name__ == '__main__':
    app.run(debug=False)