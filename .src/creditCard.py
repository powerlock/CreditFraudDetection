#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Exploration

# This project will focus on credit card fraud activity.
# Fraudulent detection is one of toughest challenges due to imbalanced data, irregular identifiable patterns, missing features, and live transactions. Creating a model with live streaming data, learn the live transaction data, update transaction pattern, and identify anomaly is pertinent in many areas. 

# # Background: Business Objectives
# It was reported that Federal Trade Commission received 2.8 million fraud reports from consumers in 2021. Consumers loss reached $5.8 billion which is 70% higher than 2020. Fraudsters are using more advanced techniques, such as machine learning, to target new customers, online transactions, and stealing identities. Currently, many models have been proposed to improve the fraud detection including KNN, logistic regression, SVM etc. For data preprocessing, data under-sampling, over-sampling, feature selection (PCA, logistic regression, SVM) have been widely used. There is report that credit card fraud detection recall can reach 0.94. However, based on the previous year’s report, fraudulent activities increase more and more. Fraudsters are using machine learning techniques to avoid defence machine learning algorithms. Simply label outliers or defining outliers are not satisfying the needs to identify attacking pattern. 
# A platform that contains data streaming, data preprocessing (feature selection, auto-labeling, grouping), model selection, model training, model relearn based on live transaction data, and prediction is highly needed. The data-model live interaction will facilitate the model selection and updating, which will further enhance the anomaly detection speed.
# 

# # Part I. Data Exploration

# In[1]:


import numpy as np
import pandas as pd



# In[2]:


filepath = "../Data/creditcard.csv"
df = pd.read_csv(filepath)


X = df.drop(['Class'], axis=1)
y = df['Class']


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 43)

cor = X_train.corr()

keep_columns = np.full(cor.shape[0], True)
for i in range(cor.shape[0] - 1):
    for j in range(i + 1, cor.shape[0] - 1):
        if (np.abs(cor.iloc[i, j]) >= 0.8): # 0.8 is the correlation threshold
            keep_columns[j] = False
selected_columns = X_train.columns[keep_columns]
X_train_reduced = X_train[selected_columns]

X_train_norm = scaler.fit_transform(X_train_reduced)
X_test_norm = scaler.fit_transform(X_test)

from supervised.automl import AutoML
automl = AutoML(algorithms=['LightGBM','Xgboost','Random Forest','Neural Network'], 
                train_ensemble=True, explain_level=2,validation_strategy={
        "validation_type": "kfold",
        "k_folds": 4,
        "shuffle": False,
        "stratify": True,
    })
automl.fit(X_train_norm, y_train)


# In[ ]:




