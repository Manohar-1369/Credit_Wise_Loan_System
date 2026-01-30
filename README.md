# Credit Wise Loan System

AI-powered loan approval prediction app built with Streamlit and a trained Logistic Regression model.

## Highlights (Resume Ready)
- Built an end-to-end supervised ML pipeline using KNN, Logistic Regression, and Naive Bayes to predict loan approval.
- Implemented binary classification with EDA, feature engineering, and model evaluation (Precision, Recall, F1).

## Project Structure
- App: app.py
- Model artifacts: logistic_regression_model.pkl, scaler.pkl, onehot_encoder.pkl, label_encoders.pkl
- Notebook: credit_wise.ipynb (training and experiments)

## How It Works
- The notebook trains the model and saves the artifacts.
- The app loads those artifacts and runs predictions from user inputs.
- DTI is collected as a ratio (0 to 1) and squared to match training features.

## Setup
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   - streamlit
   - pandas
   - numpy
   - scikit-learn
   - matplotlib

## Run the App
- From the project folder, run:
  - streamlit run app.py

## Update the Model
1. Open credit_wise.ipynb.
2. Re-train the model.
3. Export and overwrite the artifacts.
4. Restart the app.

## Deployment (Streamlit Community Cloud)
1. Push this project to GitHub.
2. Click Deploy in Streamlit.
3. Select the repo and set the main file to app.py.

## Notes
- Keep the app inputs aligned with training features.
- If you change features, re-train and re-export all artifacts.
