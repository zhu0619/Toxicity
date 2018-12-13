# Toxicity prediction
This repository contains the source code for predicting the toxicity of chemical compounds from based on SMILES strings.

# Files

- _Assessment Report_

- _data.zip_ includes the pre-processed smiles features. Please uncompress this file. 

- _toxicity_modul.py_ contains the necessary functions for toxicity prediction.

- _smiles_feature_pre-processing.py_ generates the smiles features from given dataset.

- _model_wt_training.py_ runs the traning process of the CNN model with class weighted + toxicity of the other targets as features.
python model_wt_training.py

- _model_w_training.py_ runs the traning process of the CNN model with class weighted.  
python model_w_training.py

- _model_res_training.py_ runs the traning process of the CNN model with over-sampled(minority class) dataset .
python model_res_training.py

- _parameter_tuning.py_ is script for hyperparameter tuning. 

- _predict_unknowns.py_ runs the prediction using the pre-built models.

- folder _figures_ contains pre-generated figures.
- folder _pre-built_model_ contains pre-built CNN models.
- folder _pre-computed_results_ contains the predictive result using pre-built models for the chemical compounds which the toxicity was previously unknown for 12 taegets.
