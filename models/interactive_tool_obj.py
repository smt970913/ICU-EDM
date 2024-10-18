import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

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


# Load the trained model
def load_model(model_index):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_FILES[model_index], map_location=device)
    return model


# Function to get user input and make predictions using the model
def main():
    # Create GUI window
    root = tk.Tk()
    root.geometry("600x800")
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)  # Set the initial size of the window (width x height)

    root.title("Fitted-Q-Evaluation of Extubation Failure Rate (EFR)")
    root.title("Fitted-Q-Evaluation of Extubation Failure Rate (EFR)")
    # root.title("Select Model")

    # Load and display the image at the top of the window
    img = Image.open('../image/illus_mechanical_vent.jpg')
    img = img.resize((400, 200))  # Adjust the size as needed
    img = ImageTk.PhotoImage(img)  # Update with the correct path to your image
    img_label = tk.Label(root, image=img)
    img_label.grid(row=0, column=0, columnspan=2, pady=10, sticky='n')
    img_label.image = img

    # Create model selection dropdown
    model_var = tk.StringVar(root)
    model_var.set("Select Model")
    model_dropdown = tk.OptionMenu(root, model_var, *[f"Model {i + 1}" for i in range(len(MODEL_FILES))])
    model_dropdown.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='n')

    def on_model_select():
        try:
            model_index = int(model_var.get().split()[1]) - 1
            model = load_model(model_index)
            scaler = StandardScaler()
            print("FQE Model loaded, ready for interaction.")

            # Clear window content
            for widget in root.winfo_children():
                widget.destroy()

            # Create variable labels and entry fields
            labels = [
                'Age (years)', 'Gender (Male=1, Female=0)', 'Weight (kg)', 'Heart Rate (bpm)',
                'Arterial O2 Pressure (mmHg)',
                'Hemoglobin (g/dL)', 'Arterial CO2 Pressure (mmHg)', 'Hematocrit (serum %)',
                'White Blood Cell Count (WBC, x10^9/L)', 'Chloride (serum, mEq/L)',
                'Creatinine (serum, mg/dL)', 'Glucose (serum, mg/dL)', 'Magnesium (mg/dL)', 'Sodium (serum, mEq/L)',
                'Arterial pH',
                'Inspired Oxygen Fraction (FiO2, %)', 'Arterial Base Excess (mmol/L)',
                'Blood Urea Nitrogen (BUN, mg/dL)', 'Potassium (serum, mEq/L)', 'Bicarbonate (HCO3, mEq/L)',
                'Platelet Count (x10^9/L)', 'Systolic Blood Pressure (mmHg)', 'Diastolic Blood Pressure (mmHg)',
                'Mean Blood Pressure (mmHg)',
                'Temperature (°C)', 'Oxygen Saturation (SaO2, %)', 'Glasgow Coma Scale (GCS) Score',
                'Positive End-Expiratory Pressure (PEEP, cmH2O)', 'Respiratory Rate (breaths/min)', 'Tidal Volume (mL)'
            ]
            entries = []

            for i, label_text in enumerate(labels):
                label = tk.Label(root, text=label_text)
                label.grid(row=i, column=0, padx=10, pady=5)
                entry = tk.Entry(root)
                entry.bind('<Return>', lambda event, next_entry=entry: next_entry.tk_focusNext().focus())
                entry.grid(row=i, column=1, padx=10, pady=5)
                entries.append(entry)

            def on_submit():
                try:
                    input_values = [float(entry.get()) for entry in entries]
                    if len(input_values) != 30:
                        messagebox.showerror("Input Error",
                                             "Please enter 30 variables, including the patient's gender.")
                        return

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
                    messagebox.showinfo("Model Response", f"Model Response: {response}")
                except ValueError:
                    messagebox.showerror("Input Error", "Invalid input, please ensure all values are numeric.")

            # Submit button
            submit_button = tk.Button(root, text="Submit", command=on_submit)
            submit_button.grid(row=len(labels), column=0, columnspan=2, pady=10)
        except ValueError:
            messagebox.showerror("Selection Error", "Please select a valid model.")

    # Confirm selection button
    select_button = tk.Button(root, text="Confirm Selection", command=on_model_select)
    select_button.grid(row=2, column=0, columnspan=2, pady=10, sticky='n')

    root.mainloop()


if __name__ == "__main__":
    main()