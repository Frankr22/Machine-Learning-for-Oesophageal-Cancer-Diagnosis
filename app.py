# Dependencies & Installs
import pandas as pd
import streamlit as st
import pickle

# Load data from S3 into a Pandas DataFrame (data needs to be scaled and cleaned already)
# s3_file_path = "s3://esophageal-cancer-biochem-data/Joined_df_cleaned.csv"
# df = pd.read_csv(s3_file_path)

local_file_path = "Data_Cleaned/User_Samples/users.csv"
df = pd.read_csv(local_file_path)
df.head()

# Load the trained model
with open('Models/Model_Saved/model6_LogisticRegression.pkl', 'rb') as f:
    model = pickle.load(f)
    
    # Define Streamlit app
def app():
    
    # Set app title
    st.title('Esophageal Cancer Risk Assessment app')

    # Add some text
    st.write("Welcome to our Esophageal Cancer Risk Assessment app! This app utilizes advanced machine learning algorithms to estimate your risk of developing esophageal cancer based on pre-screening and blood sample data. Esophageal cancer is a life-threatening disease affecting millions of people worldwide, and early diagnosis is crucial for improving survival rates. Traditional diagnostic methods, such as endoscopy, can be invasive and expensive. Our app aims to provide a faster, more affordable, and less invasive alternative by leveraging machine learning techniques like logistic regression, decision trees, and support vector machines. Using a dataset of biochemical data from patients with varying esophageal conditions, our models have been trained and evaluated to deliver accurate predictions. Get started by inputting your data to assess your esophageal cancer risk.")
    # st.image("Images/ai-generated-image-dalle.png", use_column_width=True)

    # Get user input
    age = st.number_input("Enter your age", value=30, min_value=18, max_value=100)
    sex = st.selectbox("Select your sex", ["male", "female"])
    bmi = st.number_input("Enter your BMI", value=25, min_value=0, max_value=50)
    diagnosed = st.selectbox("Have you been diagnosed?", ["No", "Barrett esophagus - no dysplasia", "Barrett esophagus - low dysplasia", "Barrett esophagus - high dysplasia", "Esophageal cancer"])

    gender_f = 1 if sex == "female" else 0
    gender_m = 1 if sex == "male" else 0

    patient_group = ""
    if diagnosed == "No":
        patient_group = "NSE"
    elif diagnosed == "Barrett esophagus - no dysplasia":
        patient_group = "BE"
    elif diagnosed == "Barrett esophagus - low dysplasia":
        patient_group = "BE-LGD"
    elif diagnosed == "Barrett esophagus - high dysplasia":
        patient_group = "BE-HGD"
    elif diagnosed == "Esophageal cancer":
        patient_group = "EAC"

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        "Age at Collection": [age],
        "BMI (kg/m2)": [bmi],
        "Gender_F": [gender_f],
        "Gender_M": [gender_m]
    })

    # If the user clicks the "Generate Blood Test Results" button, fetch a sample row from the dataset
    if st.button("Generate Blood Test Results"):
        sample_row = df.sample(n=1)
        blood_results_df = sample_row.drop(columns=["Patient Group", "Age at Collection", "BMI (kg/m2)", "Gender_F", "Gender_M"])
        user_input = pd.concat([user_input, blood_results_df.reset_index(drop=True)], axis=1)

        # Make a prediction using the model
        prediction = model.predict(user_input)

        # Display the prediction
        if prediction[0] == 1:
            st.write("You have a high risk of developing esophageal cancer.")
        else:
            st.write("You have a low risk of developing esophageal cancer.")
    else:
        st.write("Click the button to generate blood test results and make a prediction.")

# Run the Streamlit app
if __name__ == '__main__':
    app()