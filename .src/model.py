# load trained models, tuning threhold to find the best recall metrics.
from ast import main
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/Users/michael/Documents/MachineLearning/CreditFraudDetection/')

import tensorflow as tf
import os
import pickle as pkl
from utils import *
import pandas as pd
model_path = '../model/'
files = os.listdir(model_path)
#print(files)

def model_refine(model_file,data_file, threshold):

    model = tf.keras.models.load_model(model_path+model_file)
    #to load it
    with open("../Data/"+data_file, "rb") as f:
        preLoaddata = pkl.load(f)
    train_data, test_data, train_labels, test_labels = preLoaddata['train_data'], preLoaddata['test_data'], preLoaddata['train_labels'], preLoaddata['test_labels']
    print(model.summary())
    pred = model.predict(test_data)
    pred = (pred>threshold)
    m = test_labels.to_numpy()
    p = pred.flatten()
    scores = stats(model_file, p, m)

    dl_scores_df = pd.DataFrame(scores)
    other_score = pd.read_csv("../Data/MachineLearningSummary.csv")
    df_all = pd.concat([other_score, dl_scores_df],ignore_index=True)
    df = df_all[['model','accuracy','precision','recall','f1_score','ROC']]
    df.sort_values(by=['accuracy','precision','recall','f1_score'], inplace=True, ascending=False)
    df.to_csv("../Data/MachineLearningSummary.csv")


if __name__== "__main__":
    data_file = 'creditCard.pkl'
    threhold = 0.3
    for f in sorted(files):
        if f[-2:] == 'h5':
            print("predictions for model ",f)
            model_refine(f, data_file, threhold)
    print("refine prediction summary is done")




