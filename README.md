# CreditFraudDetection
In this project, I am exploring credit card fraud activity.

It was reported that Federal Trade Commission received 2.8 million fraud reports from consumers in 2021. Consumers loss reached $5.8 billion which is 70% higher than 2020. Fraudsters are using more advanced techniques, such as machine learning, to target new customers, online transactions, and stealing identities. Currently, many models have been proposed to improve the fraud detection including KNN, logistic regression, SVM etc. For data preprocessing, data under-sampling, over-sampling, feature selection (PCA, logistic regression, SVM) have been widely used. There is report that credit card fraud detection recall can reach 0.94. However, based on the previous year’s report, fraudulent activities increase more and more. Fraudsters are using machine learning techniques to avoid defence machine learning algorithms. Simply label outliers or defining outliers are not satisfying the needs to identify attacking pattern. 
A platform that contains data streaming, data preprocessing (feature selection, auto-labeling, grouping), model selection, model training, model relearn based on live transaction data, and prediction is highly needed. The data-model live interaction will facilitate the model selection and updating, which will further enhance the anomaly detection speed.

# Base model exploration

# Select model with autoML
XGBoost showed the best performance among other classifications models.
<p align="center">
    <img src=".src/AutoML_1/ldb_performance_boxplot.png" alt="drawing" width="400"/>
    <img src = '.src/AutoML_1/2_Default_Xgboost/cumulative_gains_curve.png'alt="drawing" width="400"/>