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

# Define Streamlit app
def app():
    # Set app title
    st.title('My Streamlit App')
    
    # Add sliders and inputs for user interaction
    age = st.slider('Age', 18, 100)
    bmi = st.slider('BMI', 10, 50)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    # Make prediction with model
    if st.button('Predict'):
        gender_binary = 1 if gender == 'Male' else 0
        prediction = model.predict([[age, bmi, gender_binary]])[0]
        
        # Display prediction
        if prediction == 0:
            st.write('Prediction: No disease')
        else:
            st.write('Prediction: Disease')
    
# Run the Streamlit app
if __name__ == '__main__':
    app()
