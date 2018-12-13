#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd

'''
python model_wt_training.py

This script runs the traning process of the CNN model with class weighted + toxicity of the other targets as features.  5-fold cross validation.

It exports the figures for all targets
- accuracy during training
- loss during training 
- the barplots of performance
- the model is also exported to .h5 file

'''

def main():
    #------------------------ load processed data -----------------------------

    print('loading preprocessed smiles feature data ...')
    processed_data = pd.read_csv(file_path +'preprocessed_data.csv',index_col=0)
    print('data loaded.')

    #------------------------ model training -----------------------------
    all_cvhistory_target = []
    all_cvscores_target = []
    for i in range(1,13):
    # for i in range(1,13):
        [cv_scores,cv_history] = toxicity_prediction_weighted_targets(processed_data,i)
        all_cvhistory_target.append(cv_history)
        all_cvscores_target.append(cv_scores)

    plot_history_sub('weighted.target',all_cvhistory_target)
    barplot_cvscores('weighted.target',all_cvscores_target,'lower right',1)

if __name__ == "__main__":
    main()

