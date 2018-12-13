# Toxicity prediction
This repository contains the source code for predicting the toxicity of chemical compounds from based on SMILES strings.

# Files

- Assessment Report
- data.zip includes the pre-processed smiles features. Please uncompress this file. 

- toxicity_modul.py contains the necessary functions for toxicity prediction.

- smiles_feature_pre-processing.py generates the smiles features from given dataset.

- model_wt_training.py runs the traning process of the CNN model with class weighted + toxicity of the other targets as features.
python model_wt_training.py

- model_w_training.py runs the traning process of the CNN model with class weighted.  
python model_w_training.py

- model_res_training.py runs the traning process of the CNN model with over-sampled(minority class) dataset .
python model_res_training.py

- predict_unknowns.py runs the prediction using the pre-built models.

- folder figures contains pre-generated figures.
- folder pre-built_model contains pre-built CNN models.
- folder pre-computed_results contains the predictive result using pre-built models for the chemical compounds which the toxicity was previously unknown for 12 taegets.
