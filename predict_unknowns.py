#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd

'''
This script 
    - generates the smiles features from given dataset 
    - loads preprocessed smiles feature and label data
    - predicts the toxicity of the chemical compounds from SMILES strings
'''


def toxicity_prediction_weighted_targets_prebuilt(feat_data,target_index, unk_smiles):
    '''
        model: class weighted  + toxicity of other targets
        input:  smile feature data ,target of interest, list of smiles
        output: export the predicted results in a csv file 
    '''
    try:
        print("toxcity prediction for target %d" % target_index)

        print('data preparation ...')
        # dataset for the corresponding taget
        X= data_prep_weighted_target(feat_data,target_index,max_row_size,unk_smiles)

        folder = 'pre-built_model/'
        my_model = load_model(folder+'target'+str(target_index)+'_model_weighted_targets.h5')
        y = my_model.predict(X)
        predictions = pd.DataFrame(y)
        predictions.index = unk_smiles
        predictions.columns = ['target'+str(target_index)]
        file_out = 'target'+str(target_index)+'_model_wt_pred.csv'
        predictions.to_csv(file_out)
        # print(predictions)
        print('Result is exported to file '+ file_out)
    except:
        print('please check your input')

def toxicity_prediction_weighted_prebuilt(feat_data,target_index,unk_smiles):
    '''
        model: class weighted
        input:  smile feature data ,target of interest, list of smiles
        output: export the predicted results in a csv file 
    '''
    try:
        print("toxcity prediction for target %d" % target_index)

        print('data preparation ...')
        # dataset for the corresponding taget
        X = data_prep_weighted(feat_data,target_index,max_row_size,unk_smiles)

        folder = 'pre-built_model/'
        my_model = load_model(folder+'target'+str(target_index)+'_model_weighted.h5')
        y = my_model.predict(X)
        predictions = pd.DataFrame(y)
        predictions.index = unk_smiles
        predictions.columns = ['target'+str(target_index)]
        file_out = 'target'+str(target_index)+'_model_w_pred.csv'
        predictions.to_csv(file_out)
        # print(predictions)
        print('Result is exported to file '+ file_out)
    except:
        print('please check your input')


def get_unknown_compounds(processed_data,target_index):
    '''
    This function returns the compound smiles which toxicity of the target of interest are unknown.
    input: processed_data smiles feature and label data
    output: The smiles strings.
    '''
    y_target_ind =processed_data.shape[1]-13+target_index
    y_out = processed_data.iloc[:,y_target_ind]
    # get the smiles  which have no target activity values
    smiles_out = y_out.index[y_out.isnull()].tolist()
    return smiles_out



def main():

    file_path = 'data/'

    #------------------------ load processed data -----------------------------

    print('loading preprocessed data ...')
    processed_data = pd.read_csv(file_path +'preprocessed_data.csv',index_col=0)
    print('data loaded.')

    # ----------- predict the toxicity for the unknown compounds --------------
    print("model: class weighted  + toxicity of other targets")
    # for i in range(1,13):
    #     unknown_smiles = get_unknown_compounds(processed_data,i)
    #     toxicity_prediction_weighted_targets_prebuilt(processed_data,i,unknown_smiles)


    # predict the toxicity for the unknown compounds 
    print("model: class weighted")
    for i in range(1,13):
        print('-----------target '+ str(i)+'--------------')
        unknown_smiles = get_unknown_compounds(processed_data,i)
        toxicity_prediction_weighted_prebuilt(processed_data,i,unknown_smiles)


if __name__ == "__main__":
    main()

