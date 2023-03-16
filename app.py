# Dependencies & Installs
import pandas as pd
import streamlit as st
import pickle

# Load data from S3 into a Pandas DataFrame (data needs to be scaled and cleaned already)
s3_file_path = "s3://esophageal-cancer-biochem-data/Joined_df_cleaned.csv"
df = pd.read_csv(s3_file_path)

# Use .pkl file to load the model
model_rf = pickle.load(open('Models/Model_Saved/model_rf_LogisticRegression.pkl', 'rb'))

# Define Streamlit app
def app():
    # Set app title
    st.title('My App')

    # Add some text
    st.write("Welcome to the Diagnosis of Esophageal Cancer app.")
    st.image("ai-generated-image-dalle.jpg", use_column_width=True)

    # Add a sidebar with some options
    sidebar_options = ["View data", "Make a prediction using only Clinical Data", "Make a prediction using only Clinical Data and Provided Lab Data"]
    sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

    # Display the data or make a prediction based on the user's selection
    if sidebar_selection == "View data":
        st.write(df.head())
    elif sidebar_selection == "Make a prediction using only Clinical Data" or sidebar_selection == "Make a prediction using only Clinical Data and Provided Lab Data":
        # Get some user input
        age = st.number_input("Enter your age", value=30, min_value=18, max_value=100)
        sex = st.selectbox("Select your sex", ["male", "female"])
        bmi = st.number_input("Enter your BMI", value=25, min_value=0, max_value=50)
        
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi]
        })
        
        # Make a prediction using the model
        prediction = model_rf.predict(user_input)
        
        # Display the prediction
        if prediction[0] == 1:
            st.write("You have a high risk of developing esophageal cancer.")
        else:
            st.write("You have a low risk of developing esophageal cancer.")
    
# Run the Streamlit app
if __name__ == '__main__':
    app()
