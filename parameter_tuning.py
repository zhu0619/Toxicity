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

def para_tuning(train_X, val_X, train_y,val_y):
    '''
    This function runs the parameter tuning.
    Resource: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
    '''
    # Create hyperparameter space
    epochs = [5,10,15]
    batches = [100,200]
    optimizers = ['rmsprop', 'adam']
    n_non_tox_samples = len(train_y[train_y==0])
    n_tox_samples = len(train_y[train_y==1])

    class_weights = [{1: n_non_tox_samples / n_tox_samples,0: 1},
                    {1: 20, 0: 1},
                    {1: 40, 0: 1}]

    # Create hyperparameter options
    hyperparameters = dict(epochs=epochs, 
                           batch_size=batches,
                           optimizer = optimizers,
                           class_weight = class_weights)

    # Wrap Keras model so it can be used by scikit-learn
    neural_network = KerasClassifier(build_fn = build_model, verbose=1)
    scores = ['recall','precision']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(estimator=neural_network, param_grid=hyperparameters,scoring='%s_macro' % score)
        clf.fit(train_X, train_y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = val_y, clf.predict(val_X)
        print(classification_report(y_true, y_pred))
        print()



def main():
    #------------------------ load processed data -----------------------------

    print('loading preprocessed smiles feature data ...')
    processed_data = pd.read_csv(file_path +'preprocessed_data.csv',index_col=0)
    print('data loaded.')

    #------------------------ model tuning -----------------------------
    img_rows, img_cols = max_row_size, len(alphabet)
    # for i in range(1,13):
    for i in range(1,2):
        print('-----------target '+ str(i)+'--------------')
        x_f,y_f = data_prep_1(processed_data,i,max_row_size)
        # prepare training and test sets
        train_X, val_X, train_y, val_y = train_test_split(x_f, y_f, stratify = y_f, test_size=0.2  )
        train_X = train_X.values.reshape(len(train_y), img_rows, img_cols, 1)
        val_X = val_X.values.reshape(len(val_y), img_rows, img_cols, 1)
        para_tuning(train_X,val_X,train_y, val_y)

if __name__ == "__main__":
    main()

