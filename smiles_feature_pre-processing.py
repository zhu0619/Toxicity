#!/usr/bin/env python
import imp
from toxicity_modul import *
import pandas as pd
import sys

'''
This script generates the smiles features from given dataset.

input:  csv file which contains the smiles string and toxicity values against targets
output: preprocessed_data.csv in folder 'data'

python smiles_feature_pre-processing.py [filename]
python smiles_feature_pre-processing.py data.scv

'''

def main():
    file_path = 'data/'

    # all characters in all SMILES strings
    alphabet = pd.read_csv(file_path +'all_alphabet.csv')['alphabet'].tolist()
    # print(alphabet)

    #--------------------- generate smiles feature data -----------------------
    # preproceess the smiles string to smiles features dataframe

    to generate feature data, please COMMENT OUT this section 
    fdata = sys.argv[1]
    home_data = pd.read_csv(file_path+fdata)

    # activity values for 12 targets
    val_tar = home_data.iloc[:,-12:]
    val_tar.index = home_data.smiles
    # remove duplicates
    val_tar = val_tar.drop_duplicates()

    # generate smiles features
    print("preprocessing smiles feature data...")
    print("This will take a while. Please be patient.")
    [all_features,all_isomeric_sml] = smiles_features_data(smiles_data)

    # combine smiles features and activity values for 12 targets
    processed_data = pd.concat([all_features,val_tar.loc[all_features.index,:]],axis=1)
    # Use the isometric smiles as row names
    processed_data.index = all_isomeric_sml
    processed_data.columns = ['Feat_'+ str(s) for s in np.arange(1,processed_data.shape[1]-11).tolist()]+ list(home_data.columns[1:])

    # export the data to csv
    processed_data.to_csv( file_path +'preprocessed_data.csv')