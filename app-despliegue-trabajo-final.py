from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, Text
from IPython.display import display
import pandas as pd
import joblib

# Load the classical model pipeline (assuming it's already loaded in the kernel)
classical_model = joblib.load('best_classical_model_pipeline.joblib')

# Load the ensemble model pipeline (assuming it's already loaded in the kernel)
ensemble_model = joblib.load('best_ensemble_model_pipeline.joblib')

# Assuming the models are already loaded in the kernel from cell -FMlqSTRL9f3

def predict_churn(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    # Create a DataFrame from the input values
    input_data = {
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }
    input_df = pd.DataFrame(input_data)

    # Add feature engineered columns if they were used in training and not handled by pipeline
    # These were commented out in a previous cell, assuming pipelines handle them.
    # If your pipelines don't add these, uncomment and add them here:
    # input_df['Contract_encoded'] = input_df['Contract'].astype('category').cat.codes
    # input_df['IsNewCustomer'] = (input_df['tenure'] == 1).astype(int)
    # input_df['CustomerValue'] = input_df['MonthlyCharges'] * input_df['tenure']

    # Ensure 'TotalCharges' is numeric and handle potential missing values (though ipywidgets should prevent this)
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df['TotalCharges'] = input_df['TotalCharges'].fillna(input_df['TotalCharges'].mean() if not input_df['TotalCharges'].isnull().all() else 0) # Handle case where all are NaN

    # Impute missing values in other columns for the single input row if necessary
    # For ipywidgets, this might not be strictly needed as inputs are controlled, but included for robustness
    for col in ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'Contract', 'PaperlessBilling', 'PaymentMethod']:
         if input_df[col].isnull().any():
             # A better approach might be to use a default value or the mode from the *training* data.
             pass

    if input_df['MonthlyCharges'].isnull().any():
         input_df['MonthlyCharges'] = input_df['MonthlyCharges'].fillna(input_df['MonthlyCharges'].mean() if not input_df['MonthlyCharges'].isnull().all() else 0)

    if 'CustomerValue' in input_df.columns and input_df['CustomerValue'].isnull().any():
         input_df['CustomerValue'] = input_df['CustomerValue'].fillna(input_df['CustomerValue'].mean() if not input_df['CustomerValue'].isnull().all() else 0)


    # Make predictions using the loaded models
    try:
        prediction_classical = classical_model.predict(input_df)
        prediction_ensemble = ensemble_model.predict(input_df)

        print("--- Prediction Results ---")
        print(f"Classical Model Prediction: {'Churn' if prediction_classical[0] == 1 else 'No Churn'}")
        print(f"Ensemble Model Prediction: {'Churn' if prediction_ensemble[0] == 1 else 'No Churn'}")
        print("--------------------------")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")


# Create interactive widgets for each feature
interact(predict_churn,
         gender=Dropdown(options=['Female', 'Male'], description='Gender:'),
         SeniorCitizen=Dropdown(options=[0, 1], description='Senior Citizen:'),
         Partner=Dropdown(options=['Yes', 'No'], description='Partner:'),
         Dependents=Dropdown(options=['Yes', 'No'], description='Dependents:'),
         tenure=IntSlider(min=0, max=72, step=1, value=1, description='Tenure (months):'),
         PhoneService=Dropdown(options=['Yes', 'No'], description='Phone Service:'),
         MultipleLines=Dropdown(options=['No phone service', 'No', 'Yes'], description='Multiple Lines:'),
         InternetService=Dropdown(options=['DSL', 'Fiber optic', 'No'], description='Internet Service:'),
         OnlineSecurity=Dropdown(options=['No', 'Yes', 'No internet service'], description='Online Security:'),
         OnlineBackup=Dropdown(options=['Yes', 'No', 'No internet service'], description='Online Backup:'),
         DeviceProtection=Dropdown(options=['No', 'Yes', 'No internet service'], description='Device Protection:'),
         TechSupport=Dropdown(options=['No', 'Yes', 'No internet service'], description='Tech Support:'),
         StreamingTV=Dropdown(options=['No', 'Yes', 'No internet service'], description='Streaming TV:'),
         StreamingMovies=Dropdown(options=['No', 'Yes', 'No internet service'], description='Streaming Movies:'),
         Contract=Dropdown(options=['Month-to-month', 'One year', 'Two year'], description='Contract:'),
         PaperlessBilling=Dropdown(options=['Yes', 'No'], description='Paperless Billing:'),
         PaymentMethod=Dropdown(options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], description='Payment Method:'),
         MonthlyCharges=FloatSlider(min=0.0, max=200.0, step=0.05, value=0.0, description='Monthly Charges:'), # Adjusted max value for better range
         TotalCharges=FloatSlider(min=0.0, max=10000.0, step=0.05, value=0.0, description='Total Charges:') # Adjusted max value for better range
        );
