# Dependencies & Installs
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image
import numpy as np

local_file_path = "Data_Cleaned/User_Samples/users.csv"
df = pd.read_csv(local_file_path)
    
# Define Streamlit app
def app():

    # Load logo image
    logo_img = Image.open("Images/ai-generated-image-dalle.png")

    # Set the app configuration
    st.set_page_config(
        page_title="CancerRisk+",
        page_icon=logo_img,
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
        'About': "Oesophageal cancer is a life-threatening disease affecting millions of people worldwide, and early diagnosis is crucial for improving survival rates. Traditional diagnostic methods, such as endoscopy, can be invasive and expensive. This app aims to provide a faster, more affordable, and less invasive alternative by leveraging machine learning. Using a dataset of biochemical data from patients with varying oesophageal conditions, the models have been trained and evaluated to deliver accurate predictions."
        }
    )

    # Expandable Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.write("Expand the sections below to access different parts of the app.")

    # Create an expandable sidebar for additional information
    with st.sidebar.expander("Learn More"):
        st.write("Find more information about oesophageal cancer, risk factors, and prevention:")
        st.write("[American Cancer Society - Oesophageal Cancer](https://www.cancer.org/cancer/esophagus-cancer.html)")
        st.write("[National Cancer Institute - Oesophageal Cancer](https://www.cancer.gov/types/esophageal)")

    # GitHub Repository section
    github_expander = st.sidebar.expander("GitHub Repository")
    with github_expander:
        st.markdown(
            "[Click here](https://github.com/Frankr22/ML-diagnosis-of-esophageal-cancer) to visit the GitHub repository for this project."
        )

    # Create an empty container for the header
    header = st.empty()
    # Add logo image and app name to the header using Markdown
    header.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{image_to_base64(logo_img)}" style="height: 50px; margin-right: 10px;" />
            <h1 style="margin: 0;">CancerRisk+</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load the trained models and scalers
    model6 = load('Models/Model_Saved/model6_LogisticRegression.joblib')
    model6_X_scaler = load('Models/Model_Saved/model6_X_scaler.joblib')
    model11 = load('Models/Model_Saved/model11_LogisticRegression.joblib')
    model11_X_scaler = load('Models/Model_Saved/model11_X_scaler.joblib')

    with st.container():
        st.write("Welcome to the Oesophageal Cancer Risk Assessment app!")
        st.write("\nThis app employs cutting-edge machine learning techniques to assess your risk of developing oesophageal cancer by analyzing pre-screening information and blood sample data.")
        st.write("\nGet started by inputting your data below to assess your oesophageal cancer risk.")

    # Add the rest of the text below the risk assessment in a separate row
    additional_text = """
    Oesophageal cancer is a life-threatening disease affecting millions of people worldwide, and early diagnosis is crucial for improving survival rates. Traditional diagnostic methods, such as endoscopy, can be invasive and expensive. Our app aims to provide a faster, more affordable, and less invasive alternative by leveraging machine learning. Using a dataset of biochemical data from patients with varying oesophageal conditions, our models have been trained and evaluated to deliver accurate predictions.
    """

    # Create two columns for the layout
    left_column, right_column = st.columns(2)

    # Get user data
    age = left_column.number_input("Enter your age:", value=30, min_value=18, max_value=100, format='%i', key='age', help='Age in years')
    sex = left_column.selectbox("Select your sex:", ["male", "female"], help='Male or Female')
    height = left_column.number_input("Enter your height:", value=170, min_value=100, max_value=250, step=1, format='%i',help='Enter in cms for Metric or inches for Imperial')
    weight = left_column.number_input("Enter your weight:", value=70, min_value=10, max_value=200, step=1, format='%i', help = 'Enter in kgs for Metric or lbs for Imperial')
    unit_system = left_column.radio("Select unit system:", options=['Metric', 'Imperial'], help='')
    diagnosed = left_column.selectbox("Have you been diagnosed with Barret esophagus?", ["No", "Barrett esophagus - no/low dysplasia"])

    # Calculate BMI based on unit system
    if unit_system == 'Metric':
        bmi = weight / ((height/100)**2) # kg / m^2
    elif unit_system == 'Imperial':
        bmi = 703 * (weight / (height**2)) # lb / in^2

    if diagnosed == "No":
        model = model6

    gender_f = 1 if sex == "female" else 0
    gender_m = 1 if sex == "male" else 0

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        "Age at Collection": [age],
        "BMI (kg/m2)": [bmi],
        "Gender_F": [gender_f],
        "Gender_M": [gender_m]
    })

    # Upload blood sample data or generate example data
    blood_sample_data = None
    uploaded_file = left_column.file_uploader("Please upload your blood sample data (CSV file) OR generate example sample data by clicking the button below", type=["csv"], help='To test the app please generate sample data. If providing blood sample data it must be in the same format as the example data. Example data can be downloaded from the GitHub repository.')
    if uploaded_file is not None:
        blood_sample_data = pd.read_csv(uploaded_file)
        left_column.success("Blood sample data uploaded successfully.")
    else:
        if left_column.button("Generate example blood sample data"):
            # Generate example blood sample data
            blood_sample_data = generate_example_data()
            left_column.success("Example blood sample data generated successfully.")

    # If the user has uploaded blood sample data, display the data
    if blood_sample_data is not None:
        left_column.write("Blood sample data:")
        left_column.dataframe(blood_sample_data)

    # Add a horizontal line and some space
    left_column.markdown("<hr/>", unsafe_allow_html=True)
    left_column.markdown("<br/>", unsafe_allow_html=True)
    
    # Select the correct model and scaler based on the user's input
    if diagnosed == "No" and blood_sample_data is None:
        model = model6
        scaler = model6_X_scaler
    elif diagnosed == "No" and blood_sample_data is not None:
        model = model6
        scaler = model6_X_scaler
    elif diagnosed == "Barrett esophagus - no/low dysplasia" and blood_sample_data is None:
        model = model11
        scaler = model11_X_scaler
    elif diagnosed == "Barrett esophagus - no/low dysplasia" and blood_sample_data is not None:
        model = model11
        scaler = model11_X_scaler

    # If the user clicks the "Generate Risk Assessment" button, scale the data and make a prediction using the model
    if left_column.button("Generate Risk Assessment"):
        # Scale the user input data
        user_input_scaled = scaler.transform(user_input)
        # If blood sample data is available, append it to the user input
        if blood_sample_data is not None:
            user_input_scaled = np.hstack([user_input_scaled, blood_sample_data.to_numpy()])
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)

        # Display the prediction
        if prediction[0] == 1:
            left_column.write(f"Based on the information you provided and our machine learning model's understanding of the relationship between various factors and oesophageal cancer risk, it is estimated that you have a higher risk of developing oesophageal cancer.\n\n The model predicts a {prediction_proba[0][1]*100:.2f}% probability of you being in the higher-risk group.\n\n Please note that this tool is not a substitute for professional medical advice, diagnosis, or treatment. The results should be considered as an estimate and should not be relied upon for decision-making regarding your health. Always consult with a healthcare professional for personalised medical advice.")
        else:
            left_column.write(f"Based on the information you provided and our machine learning model's understanding of the relationship between various factors and oesophageal cancer risk, it is estimated that you have a lower risk of developing oesophageal cancer.\n\n The model predicts a {prediction_proba[0][0]*100:.2f}% probability of you being in the lower-risk group.\n\n Please note that this tool is not a substitute for professional medical advice, diagnosis, or treatment. The results should be considered as an estimate and should not be relied upon for decision-making regarding your health. Always consult with a healthcare professional for personalised medical advice.")
    else:
        left_column.write("Click the button to generate risk assessment.")

    # Create a new row to display the additional text below the risk assessment tool
    additional_text_row = st.container()
    with additional_text_row:
        st.markdown("## About")
        st.markdown(additional_text)

def image_to_base64(img):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_example_data():
    # Use a sample row from users.csv as the example blood sample data
    sample_row = df.sample(n=1)
    blood_results_df = sample_row.drop(columns=["Patient Group", "Age at Collection", "BMI (kg/m2)", "Gender_F", "Gender_M"])
    return blood_results_df.reset_index(drop=True)

# Run the Streamlit app
if __name__ == '__main__':
    app()