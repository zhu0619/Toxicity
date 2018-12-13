# Toxicity prediction
This repository contains the source code for predicting the toxicity of chemical compounds from based on SMILES strings.

# Files

- ___Assessment Report__

- __data.zip__ includes the pre-processed smiles features. Please uncompress this file. 

- __toxicity_modul.py__ contains the necessary functions for toxicity prediction.

- __smiles_feature_pre-processing.py__ generates the smiles features from given dataset.

- ___model_wt_training.py__ runs the traning process of the CNN model with class weighted + toxicity of the other targets as features.
python model_wt_training.py

- __model_w_training.py__ runs the traning process of the CNN model with class weighted.  
python model_w_training.py

- __model_res_training.py__ runs the traning process of the CNN model with over-sampled(minority class) dataset .
python model_res_training.py

- __parameter_tuning.py__ is script for hyperparameter tuning. 

- __predict_unknowns.py__ runs the prediction using the pre-built models.

- folder __figures__ contains pre-generated figures.
- folder __pre-built_model__ contains pre-built CNN models.
- folder __pre-computed_results__ contains the predictive result using pre-built models for the chemical compounds which the toxicity was previously unknown for 12 taegets.
