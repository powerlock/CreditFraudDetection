from sklearn import metrics

def stats(modelname, predictions, labels):
  score={'model':[],'accuracy':[], 'precision':[], 'recall':[], 'f1_score':[],'ROC':[]}

  score['model'].append(modelname)
  accuracy = metrics.accuracy_score(labels, predictions)
  precision = metrics.precision_score(labels, predictions)
  recall = metrics.recall_score(labels, predictions)
  f1_score = metrics.f1_score(labels, predictions)
  if modelname in ['deepLearninig']:
    auc = None
  else:
    auc = metrics.auc(labels, predictions)
  score['accuracy'].append(accuracy)
  score['precision'].append(precision)
  score['recall'].append(recall)
  score['f1_score'].append(f1_score)
  score['ROC'].append(auc)
  print("Accuracy = {}".format(accuracy))
  print("Precision = {}".format(precision))
  print("Recall = {}".format(recall))
  
  return score