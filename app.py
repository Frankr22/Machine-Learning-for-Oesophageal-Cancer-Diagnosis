# Dependencies & Installs
import pandas as pd
import streamlit as st
from joblib import load

# local_file_path = "Data_Cleaned/User_Samples/users.csv"
# df = pd.read_csv(local_file_path)
# df.head()
    
# Define Streamlit app
def app():
    
    # Load the trained model and scaler
    model6 = load('Models/Model_Saved/model6_LogisticRegression.joblib')
    model6_X_scaler = load('Models/Model_Saved/model6_X_scaler.joblib')

    # Set app title
    st.title('Oesophageal Cancer Risk Assessment app')

    # Create two columns for the layout
    left_column, right_column = st.columns(2)

    left_column.write("Welcome to our Oesophageal Cancer Risk Assessment app!")
    left_column.write("\nThis app utilizes advanced machine learning algorithms to estimate your risk of developing oesophageal cancer based on pre-screening and blood sample data.")
    left_column.write("\nGet started by inputting your data to assess your oesophageal cancer risk.")
    left_column.image("Images/ai-generated-image-dalle.png", width=300)

    # Add the rest of the text below the risk assessment in a separate row
    additional_text = """
    Oesophageal cancer is a life-threatening disease affecting millions of people worldwide, and early diagnosis is crucial for improving survival rates. Traditional diagnostic methods, such as endoscopy, can be invasive and expensive. Our app aims to provide a faster, more affordable, and less invasive alternative by leveraging machine learning. Using a dataset of biochemical data from patients with varying oesophageal conditions, our models have been trained and evaluated to deliver accurate predictions.
    """

    # Get user input
    age = right_column.number_input("Enter your age", value=30, min_value=18, max_value=100)
    sex = right_column.selectbox("Select your sex", ["male", "female"])
    bmi = right_column.number_input("Enter your BMI", value=25, min_value=0, max_value=50)
    diagnosed = right_column.selectbox("Have you been diagnosed with Barret esophagus?", ["No", "Barrett esophagus - no dysplasia", "Barrett esophagus - low dysplasia", "Barrett esophagus - high dysplasia"])

    if diagnosed == "No":
        model = model6

    gender_f = 1 if sex == "female" else 0
    gender_m = 1 if sex == "male" else 0

    # patient_group = ""
    # if diagnosed == "No":
    #     patient_group = "NSE"
    # elif diagnosed == "Barrett esophagus - no dysplasia":
    #     patient_group = "BE"
    # elif diagnosed == "Barrett esophagus - low dysplasia":
    #     patient_group = "BE-LGD"
    # elif diagnosed == "Barrett esophagus - high dysplasia":
    #     patient_group = "BE-HGD"

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        "Age at Collection": [age],
        "BMI (kg/m2)": [bmi],
        "Gender_F": [gender_f],
        "Gender_M": [gender_m]
    })

    # Scale the user input data
    user_input_scaled = model6_X_scaler.transform(user_input)
    
    # If the user clicks the "Generate Risk Assessment" button, make a prediction using the model
    if right_column.button("Generate Risk Assessment"):
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)

        # Display the prediction
        if prediction[0] == 1:
            right_column.write(f"Based on the information you provided and our machine learning model's understanding of the relationship between various factors and oesophageal cancer risk, it is estimated that you have a higher risk of developing oesophageal cancer. The model predicts a {prediction_proba[0][1]*100:.2f}% probability of you being in the higher-risk group.\n\nPlease note that this tool is not a substitute for professional medical advice, diagnosis, or treatment. The results should be considered as an estimate and should not be relied upon for decision-making regarding your health. Always consult with a healthcare professional for personalised medical advice.")
        else:
            right_column.write(f"Based on the information you provided and our machine learning model's understanding of the relationship between various factors and oesophageal cancer risk, it is estimated that you have a lower risk of developing oesophageal cancer. The model predicts a {prediction_proba[0][0]*100:.2f}% probability of you being in the lower-risk group.\n\nPlease note that this tool is not a substitute for professional medical advice, diagnosis, or treatment. The results should be considered as an estimate and should not be relied upon for decision-making regarding your health. Always consult with a healthcare professional for personalised medical advice.")
    else:
        right_column.write("Click the button to generate risk assessment.")

    # # If the user clicks the "Generate Blood Test Results" button, fetch a sample row from the dataset
    # if st.button("Generate Blood Test Results"):
    #     sample_row = df.sample(n=1)
    #     blood_results_df = sample_row.drop(columns=["Patient Group", "Age at Collection", "BMI (kg/m2)", "Gender_F", "Gender_M"])
    #     user_input = pd.concat([user_input, blood_results_df.reset_index(drop=True)], axis=1)

    # Create a new row to display the additional text below the risk assessment tool
    additional_text_row = st.container()
    with additional_text_row:
        st.markdown("## About")
        st.markdown(additional_text)

# Run the Streamlit app
if __name__ == '__main__':
    app()