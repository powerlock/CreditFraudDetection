# CreditFraudDetection
In this project, I am exploring credit card fraud activity.

It was reported that Federal Trade Commission received 2.8 million fraud reports from consumers in 2021. Consumers loss reached $5.8 billion which is 70% higher than 2020. Fraudsters are using more advanced techniques, such as machine learning, to target new customers, online transactions, and stealing identities. Currently, many models have been proposed to improve the fraud detection including KNN, logistic regression, SVM etc. For data preprocessing, data under-sampling, over-sampling, feature selection (PCA, logistic regression, SVM) have been widely used. There is report that credit card fraud detection recall can reach 0.94. However, based on the previous year’s report, fraudulent activities increase more and more. Fraudsters are using machine learning techniques to avoid defence machine learning algorithms. Simply label outliers or defining outliers are not satisfying the needs to identify attacking pattern. 

A platform that contains data streaming, data preprocessing (feature selection, auto-labeling, grouping), model selection, model training, model relearn based on live transaction data, and prediction is highly needed. The data-model live interaction will facilitate the model selection and updating, which will further enhance the anomaly detection speed.

In the first step, I will explore different classfication models with the same set of data. By comparing their performance, hopefully, the best model will be selected for the task.

In the second step, I will explore GAN simulation data as live transaction data to feed the model for prediction. And use monitoring service for the model.

# Base model exploration
## Data exploration
* Since the data is highly imbalanced, the preprocessing of the data is needed. In the current steps, undersampling method is explored for the model selection. Also, in LogisticRegression, 'balanced' weight is used for the model.
* The nature of the data provides an unique way for auto-encoder model. Basically, data 
* Explore correlation, remove highly correlated data.
* Understand the data trend with visulization tools.

## Select model with autoML for several classification models
* Models include LightGBM','Xgboost','Random Forest','Neural Network'.
* XGBoost showed the best performance among other classifications models.
<p align="center">
    <img src=".src/AutoML_1/ldb_performance_boxplot.png" alt="drawing" width="400"/>
    <img src = '.src/AutoML_1/2_Default_Xgboost/cumulative_gains_curve.png'alt="drawing" width="400"/>

## Use autoencoder model to do the test
* Deep learning with dense layer.
* Difference between Standard Scaler and minMax scaler. The former has large loss around 2.7 and it hardly decreases in the first 30 epochs. The latter gives smaller loss 
* add dropout layer to solve overfitting problem.
* refer to keras.loss, we can identify that binary_crossentropy is better than mae, since the latter is better for regression. When using binary_crossentropy, we should normalize the data in the range [0,1], that is MinMaxScaler insead of standard scaler.
* outliers in the data will cause train the test data huge difference.

## SMOTE synthetic minority oversampling 


# Summary of various models performance:

,model,| accuracy, | precision,| recall,| f1_score,| ROC
----------------------------------------------------------
0,"XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              class_weight='balanced', colsample_bylevel=1, colsample_bynode=1,
              colsample_bytree=1, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_to_onehot=4, max_delta_step=0, max_depth=6,
              max_leaves=0, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=0,
              num_parallel_tree=1, predictor='auto', random_state=42,
              reg_alpha=0, ...)",0.999490889,0.942528736,0.773584906,0.849740933,
1,"RandomForestClassifier(class_weight='balanced', random_state=42)",0.999420666,0.929411765,0.745283019,0.827225131,
2,model_v10_conv1D.h5,0.9992275497879272,0.78125,0.7874015748031497,0.7843137254901962,
18,model_v9_conv1D.h5,0.9991432824920649,0.83,0.6535433070866141,0.7312775330396475,
3,deepLearninig,0.999101149,0.752,0.74015748,0.746031746,
4,model_v12_conv1D.h5,0.9991011488441336,0.8461538461538461,0.6062992125984252,0.7064220183486238,
5,model_v8_conv1D.h5,0.9990871042948232,0.8875,0.5590551181102362,0.6859903381642511,
6,model_v11_conv1D.h5,0.9990730597455127,0.8674698795180723,0.5669291338582677,0.6857142857142857,
7,model_v5.h5,0.9990028369989608,0.7916666666666666,0.5984251968503937,0.6816143497757847,
8,model_v1.h5,0.9989185697030982,0.6582278481012658,0.8188976377952756,0.7298245614035088,
9,model_v7.h5,0.998806213308615,0.8181818181818182,0.4251968503937008,0.5595854922279793,
10,model_v3.h5,0.998778124209994,0.7777777777777778,0.4409448818897638,0.5628140703517588,
11,model_v4.h5,0.9984831886744754,0.5568862275449101,0.7322834645669292,0.6326530612244898,
12,"SVC(class_weight={0: 1, 1: 100})",0.998419999,0.556338028,0.745283019,0.637096774,0.500052765
13,model_v2.h5,0.9971208673913654,0.3636363636363636,0.8188976377952756,0.5036319612590798,
14,model_v6.h5,0.9970646891941236,0.2747252747252747,0.3937007874015748,0.3236245954692556,
15,SVC(class_weight='balanced'),0.996839999,0.341880342,0.754716981,0.470588235,0.500052765
16,"LogisticRegression(class_weight='balanced', max_iter=100000, random_state=42)",0.977704435,0.07079646,0.905660377,0.131326949,0.63849188
17,autoencoder,0.167234297,0.001600404,0.76,0.003194083,141.0
