#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd

'''
python model_w_training.py

This script runs the traning process of the CNN model with class weighted.  5-fold cross validation.

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
    all_cvhistory_weighted = []
    all_cvscores_weighted = []
    # for i in range(1,13):
        [cv_scores,cv_history] = toxicity_prediction_weighted(processed_data,i)
        all_cvhistory_weighted.append(cv_history)
        all_cvscores_weighted.append(cv_scores)
        gc.collect()
    
    plot_history_sub('weighted',all_cvhistory_weighted)
    barplot_cvscores('weighted',all_cvscores_weighted, 'lower center' ,4)


if __name__ == "__main__":
    main()

