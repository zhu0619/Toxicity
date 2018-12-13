#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd



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

