# Dependencies & Installs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a SparkSession
spark = SparkSession.builder.appName("myApp").getOrCreate()

# # Load data from S3 into a Spark DataFrame (data needs to be scaled and cleaned already)
# s3_file_path = "s3a://my-bucket/Joined_df_cleaned.csv"
# df = spark.read.format("csv").option("header", True).load(s3_file_path)

# Load data
joined_df = pd.read_csv('Data_Cleaned/Joined_df_cleaned.csv')

# Drop the Unnamed: 0 column from joined_df using iloc
joined_df = joined_df.iloc[:, 1:]
# Create a StandardScaler object
scaler = StandardScaler()
# Select only the columns that need to be scaled
columns_to_scale = joined_df.columns[1:]
# Scale the selected columns
scaled_columns = scaler.fit_transform(joined_df[columns_to_scale])
# Create a new dataframe with the scaled columns
scaled_df = pd.concat([joined_df['Patient Group'], pd.DataFrame(scaled_columns, columns=columns_to_scale)], axis=1)
# Create a new column that maps the Group column values to either BE-HGD or EAC
scaled_df['target'] = joined_df['Patient Group'].map({'BE-HGD': 0, 'EAC': 1, 'BE': 0, 'BE-ID': 0, 'BE-LGD': 0, 'NSE': 0})
# Drop the Group column as it is no longer needed
scaled_df.drop('Patient Group', axis=1, inplace=True)

## USE .h5 Model Instead

# Convert Spark DataFrame to Pandas DataFrame
pdf = df.toPandas()

# Split the dataset into training and testing sets
X = scaled_df.drop(columns=['target'])
y = scaled_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the best hyperparameters
best_params = {'C': 0.1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}

# Create a Logistic Regression model with the best hyperparameters
model = LogisticRegression(**best_params)
# Train the model on the training data
model.fit(X_train, y_train)

# Define a function to make predictions using the model
def predict(model, data):
    return model.predict(data)

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
