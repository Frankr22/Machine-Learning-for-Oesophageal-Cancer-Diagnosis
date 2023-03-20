# Machine Learning for Oesophageal Cancer Diagnosis

<figure>
  <img src="Images/ai-generated-image-dalle.png" alt="DALL-E AI image" width="25%">
</figure>

This project aims to develop a data product prototype, using machine learning algorithms and inputs from clinical and biochemical data, to diagnose oesophageal cancer.

## Background
Oesophageal cancer is 8th most common cancer in the world, and is the 6th most common cause of cancer-related deaths (WHO). Ealy diagnosis is crucial for timely treatment and improved survival rates. Traditional diagnostic methods, such as endoscopy, can be invasive and expensive. This app aims to provide a faster, more affordable, and less invasive alternative by leveraging machine learning. Using a dataset of biochemical data from patients with varying oesophageal conditions, the models have been trained and evaluated to deliver accurate predictions.

## Dataset
We downloaded datasets in csv files from the supplementary data of a published peer-reviewed journal article that included 2 cohorts: 48-sample American cohort and 252-sample Australian cohort (doi:10.1074/mcp.RA118.000734)
The project uses clinical data (eg. BMI, sex, age) and biochemical data (protein levels) of 300 people, who can be Healthy, have Barrett's Oesophagus or have been diagnosed with Oesophageal cancer. 

## Machine Learning Algorithms
The project uses several machine learning algorithms to predict the risk of having oesophageal cancer in individuals, including logistic regression, decision trees, and support vector machines. 
The algorithms are trained on the dataset and evaluated using various performance metrics, such as accuracy, precision, and recall. We will also investigate whether unsupervised machine learning or deep learning can be used to predict disease categories and if it performs better or worse than supervised machine learning.

## Usage
- Visit our website for the Cancer Risk Assessment App: [cancer-risk.streamlit.app](https://cancer-risk.streamlit.app/)
- To run the streamlit app locally, run code `streamlit run app.py` in GitBash, same directory as `app.py`
- To explore the backend of the project, you will need to have Python and several Python packages installed, including Scikit-learn, NumPy, and Pandas (see requirements.txt). You can clone the repository and run the scripts provided to preprocess the data, train and evaluate the machine learning algorithms, and make predictions on new data. You can also explore the Jupyter notebooks provided to visualize the data and results.

## Repository Structure
Data/ directory contains the dataset
- `.streamlit/` directory contains streaming configuration
- `Analysis_Notebooks/` directory contains jupyter notebooks used in Data Analysis
- `Data_Cleaned/` contains CSV files of cleaned data
- `Data_Source/` contains CSV files of original data source
- `Documents/` contains documentation
- `ETL_Notebooks/` directory contains jupyter notebooks used in Extract, Transform and Load pre-processing
- `Images/` directory contains images used in the project such as visualisations.
- `Models/` directory contains images used in the project such as visualisations.

## Technologies Used
We utilized the following technologies in our project:
- Scikit-learn for machine learning
- Python Pandas for data manipulation
- Tableau for data visualizations
- Streamlit for front-end web development

## Future Work
- Increase the size of the training dataset
- Improve Data storage
- Deploy the web application on a cloud platform, such as Google Cloud or Amazon AWS
- Explore unsupervised learning and deep learning models

## Contributors
This project was developed by Robert Franklin (GitHub username: Frankr22), Marisa Duong (GitHub username: MarDuo2022), Brianna O'Connor (GitHub username: borruu) as part of their capstone project for their data analytics course. 
We welcome contributions from other data scientists, machine learning enthusiasts, and medical professionals who are interested in improving the diagnosis of oesophageal cancer.

## References
- Data Source: Shah AK, et al. (2018) Evaluation of Serum Glycoprotein Biomarker Candidates for Detection of Esophageal Adenocarcinoma and Surveillance of Barrett's Esophagus. Mol Cell Proteomics 17(12):2324-2334. doi:10.1074/mcp.RA118.000734
- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
- Python Pandas documentation: https://pandas.pydata.org/docs/
- Tableau documentation: https://help.tableau.com/current/guides/get-started-tutorial/en-us/get-started-tutorial-home.htm
- Streamlit documentation: https://docs.streamlit.io/en/stable/
