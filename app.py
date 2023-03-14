# Dependencies & Installs
# import pandas as pd
# import numpy as np
# import warnings
import streamlit as st
# import pandas as pd
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col

# # Create a SparkSession
# spark = SparkSession.builder.appName("myApp").getOrCreate()

# # # Load data from S3 into a Spark DataFrame (data needs to be scaled and cleaned already)
# # s3_file_path = "s3a://my-bucket/Joined_df_cleaned.csv"
# # df = spark.read.format("csv").option("header", True).load(s3_file_path)

# # Convert Spark DataFrame to Pandas DataFrame
# pdf = df.toPandas()

## USE .h5 Model Instead
# # Split the dataset into training and testing sets
# X = scaled_df.drop(columns=['target'])
# y = scaled_df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Define the best hyperparameters
# best_params = {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
# # Create a Logistic Regression model with the best hyperparameters
# model = LogisticRegression(**best_params)
# # Train the model on the training data
# model.fit(X_train, y_train)
# # Define a function to make predictions using the model
# def predict(model, data):
#     return model.predict(data)

# Define Streamlit app
def app():
    # Set app title
    st.title('My App')

    # Add some text
    st.write("This is my app.")

    # Add a sidebar with some options
    sidebar_options = ["View data", "Make a prediction using only Clinical Data", "Make a prediction using only Clinical Data and Provided Lab Data"]
    sidebar_selection = st.sidebar.selectbox("Select an option", sidebar_options)

    # Display the data or make a prediction based on the user's selection
    if sidebar_selection == "View data":
        st.write(pdf.head())
    elif sidebar_selection == "Make a prediction":
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
        prediction = predict(model, user_input)
        
        # Display the prediction
        if prediction[0] == 1:
            st.write("You have a high risk of developing a disease.")
        else:
            st.write("You have a low risk of developing a disease.")
    
# Run the Streamlit app
if __name__ == '__main__':
    app()
