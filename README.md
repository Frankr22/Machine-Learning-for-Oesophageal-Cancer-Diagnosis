# Machine Learning Esophageal Cancer Diagnosis

<img src="https://user-images.githubusercontent.com/110333750/222064989-03d22a6e-e69f-4e54-9aa4-ad27341edb0d.png" alt="DALL·E 2023-03-01 14 47 40 - Can machine learning be used to diagnose cancer?" width="25%">
![image description](image-link "An AI-generated image based on the project research question, created with the help of DALL-E.")

This project aims to investigate the potential of machine learning algorithms in diagnosing esophageal cancer using biochemical data. The project focuses on using raw data output from biochemical labs to predict the presence of esophageal cancer in patients.

## Background
Esophageal cancer is a deadly disease that affects millions of people worldwide. Early diagnosis is key to increasing survival rates, but traditional diagnostic methods such as endoscopy can be invasive and costly. This project aims to explore the potential of machine learning algorithms to diagnose esophageal cancer using raw biochemical data, which could potentially lead to a faster, cheaper, and less invasive diagnosis.

## Dataset
The project uses a dataset of biochemical data from patients with healthy esophagi, Barrett's esophagus, and esophageal cancer. For this project, we will focus on using the cohort of 253 samples as a training dataset and the cohort of 45 samples as a validation dataset. Additionally, we might try shuffling the cohort samples to create mixed training and validation datasets to evaluate if the models developed using this approach perform better.

## Machine Learning Algorithms
The project uses several machine learning algorithms to predict the presence of esophageal cancer in patients, including logistic regression, decision trees, and support vector machines. The algorithms are trained on the dataset and evaluated using various performance metrics, such as accuracy, precision, and recall. We will also investigate whether unsupervised machine learning or deep learning can be used to predict disease categories and if it performs better or worse than supervised machine learning.

## Usage
To use this project, you will need to have Python and several Python packages installed, including Scikit-learn, NumPy, and Pandas. You can clone the repository and run the scripts provided to preprocess the data, train and evaluate the machine learning algorithms, and make predictions on new data. You can also explore the Jupyter notebooks provided to visualize the data and results.

## Project Structure
Data/ directory contains the dataset
- `Images/` directory contains images used in the project such as visualisations.

## References
- Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
- Bootstrap documentation: https://getbootstrap.com/docs/5.1/getting-started/introduction/
- Plotly documentation: https://plotly.com/javascript/
- Leaflet documentation: https://leafletjs.com/

## Contributors
This project was developed by Robert Franklin (GitHub username: Frankr22), Marisa Duong (GitHub username: MarDuo2022), Brianna O'Connor (GitHub username: borruu) as part of their capstone project for their data analytics course. They welcome contributions from other data scientists, machine learning enthusiasts, and medical professionals who are interested in improving the diagnosis of esophageal cancer.
