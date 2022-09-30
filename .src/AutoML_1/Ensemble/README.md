# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model             |   Weight |
|:------------------|---------:|
| 2_Default_Xgboost |        1 |

## Metric details
|           |      score |     threshold |
|:----------|-----------:|--------------:|
| logloss   | 0.00254644 | nan           |
| auc       | 0.973061   | nan           |
| f1        | 0.275587   |   0.00244505  |
| accuracy  | 0.992017   |   0.00244505  |
| precision | 0.162824   |   0.00244505  |
| recall    | 1          |   1.61615e-05 |
| mcc       | 0.380157   |   0.00244505  |


## Metric details with threshold from accuracy metric
|           |      score |    threshold |
|:----------|-----------:|-------------:|
| logloss   | 0.00254644 | nan          |
| auc       | 0.973061   | nan          |
| f1        | 0.275587   |   0.00244505 |
| accuracy  | 0.992017   |   0.00244505 |
| precision | 0.162824   |   0.00244505 |
| recall    | 0.896373   |   0.00244505 |
| mcc       | 0.380157   |   0.00244505 |


## Confusion matrix (at threshold=0.002445)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |           225680 |             1779 |
| Labeled as 1 |               40 |              346 |

## Learning curves
![Learning curves](learning_curves.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



[<< Go back](../README.md)
