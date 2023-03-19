# Machine Learning Oesophageal Cancer Diagnosis

<figure>
  <img src="Images/ai-generated-image-dalle.png" alt="DALL-E AI image" width="25%">
</figure>

This project aims to investigate the potential of machine learning algorithms in diagnosing oesophageal cancer using biochemical data. The project focuses on using raw data output from biochemical labs to predict the presence of oesophageal cancer in patients.

## Table of Contents

## Background
Oesophageal cancer is a deadly disease that affects millions of people worldwide. Early diagnosis is key to increasing survival rates, but traditional diagnostic methods such as endoscopy can be invasive and costly. This project aims to explore the potential of machine learning algorithms to diagnose oesophageal cancer using raw biochemical data, which could potentially lead to a faster, cheaper, and less invasive diagnosis.

## Dataset
The project uses a dataset of biochemical data from patients with healthy esophagi, Barrett's esophagus, and oesophageal cancer. For this project, we will focus on using the cohort of 253 samples as a training dataset and the cohort of 45 samples as a validation dataset. Additionally, we might try shuffling the cohort samples to create mixed training and validation datasets to evaluate if the models developed using this approach perform better.

## Machine Learning Algorithms
The project uses several machine learning algorithms to predict the presence of esophageal cancer in patients, including logistic regression, decision trees, and support vector machines. The algorithms are trained on the dataset and evaluated using various performance metrics, such as accuracy, precision, and recall. We will also investigate whether unsupervised machine learning or deep learning can be used to predict disease categories and if it performs better or worse than supervised machine learning.

## Usage
- Visit our website for the Cancer Risk Assessment App: [cancer-risk.streamlit.app](https://cancer-risk.streamlit.app/)
- To explore the backend of the project, you will need to have Python and several Python packages installed, including Scikit-learn, NumPy, and Pandas (see requirements.txt). You can clone the repository and run the scripts provided to preprocess the data, train and evaluate the machine learning algorithms, and make predictions on new data. You can also explore the Jupyter notebooks provided to visualize the data and results.

## Project Structure
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
- Streamlit for frontend web development

## Future Work
- Explore unsupervised learning and deep learning models to imporve the model
- Increase the size of the dataset
- Deploy the web application on a cloud platform, such as Google Cloud or Amazon AWS

## Contributors
This project was developed by Robert Franklin (GitHub username: Frankr22), Marisa Duong (GitHub username: MarDuo2022), Brianna O'Connor (GitHub username: borruu) as part of their capstone project for their data analytics course. They welcome contributions from other data scientists, machine learning enthusiasts, and medical professionals who are interested in improving the diagnosis of esophageal cancer.

## References
- Data Source: Shah AK, et al. (2018) Evaluation of Serum Glycoprotein Biomarker Candidates for Detection of Esophageal Adenocarcinoma and Surveillance of Barrett's Esophagus. Mol Cell Proteomics 17(12):2324-2334. doi:10.1074/mcp.RA118.000734
- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
- Python Pandas documentation: https://pandas.pydata.org/docs/
- Tableau documentation: https://help.tableau.com/current/guides/get-started-tutorial/en-us/get-started-tutorial-home.htm
- Streamlit documentation: https://docs.streamlit.io/en/stable/
