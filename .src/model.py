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
model_path = '../model/trainining/'
files = os.listdir(model_path)

def model_refine(model_file,data_file, threshold):
    model = tf.load(model_file)
    #to load it
    with open("../Data/"+data_file, "rb") as f:
        preLoaddata = pkl.load(f)
    train_data, test_data, train_labels, test_labels = preLoaddata['train_data'], preLoaddata['test_data'], preLoaddata['train_labels'], preLoaddata['test_labels']

    pred = model.predict(test_data, test_labels)
    pred = [pred>threshold]
    m = test_labels.to_numpy()
    p = pred.flatten()
    scores = stats(model, p, m)

    dl_scores_df = pd.DataFrame(scores)
    other_score = pd.read_csv("../Data/MachineLearningSummary.csv")
    df_all = pd.concat([other_score, dl_scores_df])
    df_all.to_csv("../Data/MachineLearningSummary.csv")

if __name__==main():
    data_file = 'creditCard.pkl'
    for f in files:
        model_refine(f, data_file, 0.3)




