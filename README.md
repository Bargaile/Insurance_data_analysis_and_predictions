
# Insurance_data_analysis_and_predictions
# Trying to predict possible Covid-19 insurance buyers

# PURPOSE

Tour & Travels Company wants to offer travel insurance package with covid-19 coverage to some and later maybe to all of their customers.
The Company requires to know (predict) which of their customers would have interest to buy this insurance package. It can be predicted based on Companies database history. In 2019 this type of insurance was offered to some segment of the customers and Company has extracted the Performance/Sales of the travel insurance package with coverage from covid-19 during that period. Company need a tool, which could 'for see' if certain customer could be interested in the travel insurance package with covid-19 coverage and in that case to offer it him/her. The aim was to find basic patterns in given data and make a tool for basic predictions of interest in covid-19 insurance.


# STRUCTURE:

- **EDA.ipynb**
Deep analysis of the given insurance data set, data cleaning, feature engineering to prepare the final, clean data set for modeling. Inferential statistical analysis was proceeded. Visualizations made with **Plotly, Seaborn, Matplotlib**. Conclusions: which features are the most promising for modeling.

- **modeling.ipynb**

In this part I tried to predict: will a person buy travel insurance with covid-19 coverage or not. As the dataset was imbalanced, the success measure for the model was chosen F1 score. Different model were used as base line, to see which of them "works" best on this data set, then according to F1 score, SVM, KNN and Random Forest classifiers were chosen for further development (different features were selected and fed to these models, **SMOTE** up sampling  and under sampling were tried, hyperparameter tuning was proceeded).

- **my_functions[folder]**

Contain .py file with all functions, used in the EDA and Modeling.
