#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from pandasql import sqldf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn import metrics

# read the data for training
train = pd.read_csv("C:\\Users\\yasha\\Downloads\\train_ctrUa4K (1).csv")
sns.set_theme(style="darkgrid")
print(train.head())

# data distribution using rows and columns
rows, columns = train.shape
print("Rows: ", rows)
print("Columns: ", columns)
print(train.info())

# to understand statistic variation
print(train.describe())


# converting characterize data into numeric
train['Gender'] = train['Gender'].replace(['Male', 'Female'], [1, 0])
train['Loan_Status'] = train['Loan_Status'].replace(['Y', 'N'], [1, 0])
train['Married'] = train['Married'].replace(['Yes', 'No'], [1, 0])
train['Education'] = train['Education'].replace(['Graduate', 'Not Graduate'], [1, 0])
train['Self_Employed'] = train['Self_Employed'].replace(['Yes', 'No'], [1, 0])


# check missing values
print(train.isnull().sum())

# fill the missing value with mode
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
train['Self_Employed'] = train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mode()[0])
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0])
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mode()[0])

# check the null blocks
train.isnull().sum()
print(train.isnull().sum())

# check columns
print(train.columns)

# check data distribution
train.hist(figsize=(15, 15))
plt.show()

# uni-variate analysis for Applicant Income
plt.figure(figsize=(15, 5))
sns.barplot(x=train['Loan_Status'], y=train['ApplicantIncome'])
#plt.xticks()

# uni-variate analysis for Co-applicant Income
plt.figure(figsize=(15, 5))
sns.barplot(x=train['Loan_Status'], y=train['CoapplicantIncome'])
#plt.xticks()

# uni-variate analysis for loan amount
plt.figure(figsize=(15, 5))
sns.barplot(x=train['Loan_Status'], y=train['LoanAmount'])
#plt.xticks()

# uni-variate analysis for loan amount term
plt.figure(figsize=(15, 5))
sns.barplot(x=train['Loan_Status'], y=train['Loan_Amount_Term'])
#plt.xticks()

# uni-variate analysis for Credit History
plt.figure(figsize=(15, 5))
sns.barplot(x=train['Loan_Status'], y=train['Credit_History'])
#plt.xticks()

# check count of target variable
print(train["Loan_Status"].value_counts())

# data analyzing using box plot method
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=train)
#plt.show()
sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=train)
#plt.show()
sns.boxplot(x="Loan_Status", y="LoanAmount", data=train)
#plt.show()
sns.boxplot(x="Loan_Status", y="Loan_Amount_Term", data=train)
#plt.show()
sns.boxplot(x="Loan_Status", y="Credit_History", data=train)
#plt.show()

# plot correlation
#cols = ["Loan_ID", "Gender", "Married",	"Dependents", "Education", "Self_Employed", "Property_Area"]
cols = ["Loan_ID", "Self_Employed",  "Education", "Property_Area", "CoapplicantIncome", "Loan_Amount_Term"]
train['Loan_Status'] = train['Loan_Status'].replace(['Y', 'N'], [1, 0])
train['Gender'] = train['Gender'].replace(['Male', 'Female'], [1, 0])
train['Married'] = train['Married'].replace(['Yes', 'No'], [1, 0])
train['Dependents'] = train['Dependents'].replace(['0', '1', '2', '3+'], [0, 1, 2, 3])
'''train['Education'] = train['Education'].replace(['Graduate', 'Not Graduate'], [1, 0])
train['Property_Area'] = train['Property_Area'].replace(['Rural', 'Semiurban', 'Urban'], [2, 1, 0])'''
train = train.drop(columns=cols, axis=1)
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = train.corr()
corr_heatmap = sns.heatmap(corr_matrix, cmap="flare", annot=True, ax=ax, annot_kws={"size": 14})
plt.show()

# plot categorical features
def categorical_valcount_hist(feature):
    test = pd.read_csv("C:\\Users\\yasha\\Downloads\\test_lAUu6dG.csv")
    print(test[feature].value_counts())
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.countplot(x=feature, ax=ax, data=test)
    plt.show()
#categorical_valcount_hist("Loan_Status")

# splitting data into train and test split
x = train.drop("Loan_Status", axis=1)
y = train["Loan_Status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7)

# Random Forest Model
# train the model
rf_clf = RandomForestClassifier(criterion='gini', bootstrap=True, random_state=100)#gini index is used for memory management
smote_sampler = SMOTE(random_state=9) #SMOTE for over sampling
pipeline = Pipeline(steps=[['smote', smote_sampler], ['classifier', rf_clf]])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# check accuracy
print("-------------------------TEST SCORES-----------------------")
print(f"Recall: { round(recall_score(y_test, y_pred)*100, 4) }")
print(f"Precision: { round(precision_score(y_test, y_pred)*100, 4) }")
print(f"F1-Score: { round(f1_score(y_test, y_pred)*100, 4) }")
print(f"Accuracy score: { round(accuracy_score(y_test, y_pred)*100, 4) }")
print(f"AUC Score: { round(roc_auc_score(y_test, y_pred)*100, 4) }")


# plotting of confusion matrix
actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
cm_display.plot()
plt.show()


print(train.columns)

# testing model
# reading the testing file
test = pd.read_csv("C:\\Users\\yasha\\Downloads\\test_lAUu6dG.csv")
cols = ["Loan_ID", "Self_Employed",  "Education", "Property_Area", "CoapplicantIncome", "Loan_Amount_Term"]
test['Gender'] = test['Gender'].replace(['Male', 'Female'], [1, 0])
test['Married'] = test['Married'].replace(['Yes', 'No'], [1, 0])
test['Dependents'] = test['Dependents'].replace(['0', '1', '2', '3+'], [0, 1, 2, 3])
test = test.drop(columns=cols, axis=1)

# cleaning of data
print("******************* Data Before Cleaning *********************")
print(test.isnull().sum())
test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0])
test['LoanAmount'] = test['LoanAmount'].fillna(test['LoanAmount'].mode()[0])
test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mode()[0])

print("******************* Data After Cleaning *********************")
print(test.isnull().sum())

# testing file using model
y_prediction = pipeline.predict(test)

# finding probability of prediction
total_y = 0
total_n = 0
for i in range(len(y_prediction)):
    if y_prediction[i] == 1:
        total_y += 1
    elif y_prediction[i] == 0:
        total_n += 1

print("Total Number of Applicants = ", len(y_prediction))
print("Total Yes Count = ", total_y)
print("Total No Count = ", total_n)
print("Probability of Yes = ", total_y/len(y_prediction))

test["Predicted_Status"] = y_prediction
print(test["Predicted_Status"])
print(type(test["Predicted_Status"]))
print(test)
test.to_csv('sample1.csv')


# In[ ]:




