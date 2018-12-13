#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd
'''
This script runs the traning process of the CNN model with class weighted + toxicity of the other targets as features.  5-fold cross validation.

It exports 
- the models are exported to target*_model_resampled.h5 
- figures accuracy during training
- figures loss during training 
- figures the barplots of performance the figures for all targets

# training for all targets
python model_res_training.py

'''

def main():
    #------------------------ load processed data -----------------------------

    print('loading preprocessed smiles feature data ...')
    processed_data = pd.read_csv(file_path +'preprocessed_data.csv',index_col=0)
    print('data loaded.')

    #--------------------------- model training -------------------------------
    all_cvhistory_resampled = []
    all_cvscores_resampled = []
    
    for i in range(1,13):
        print('-----------target '+ str(i)+'--------------')
        [cv_scores,cv_history] = toxicity_prediction_resampled(processed_data,i)
        all_cvhistory_resampled.append(cv_history)
        all_cvscores_resampled.append(cv_scores)
        # gc.collect()
    plot_history_sub('resampled',all_cvhistory_resampled)
    barplot_cvscores('resampled',all_cvscores_resampled, 'lower right' ,1)


if __name__ == "__main__":
    main()



