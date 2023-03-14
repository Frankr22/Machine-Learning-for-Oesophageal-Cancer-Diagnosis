import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs')

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
