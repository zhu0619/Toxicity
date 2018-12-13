#!/usr/bin/env python
__author__ = "Lu Zhu"
__email__ = "zhu.lu@hotmail.com"

'''
This script contains the necessary functions for
- data processing and preparation
- CNNs construction 
- evaluation 
- figure generation
'''


# Several helpful packages to load in 

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPool2D
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,train_test_split,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,average_precision_score,roc_curve,auc,fbeta_score
import pubchempy as pcp
import imblearn
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import itertools as it

# Input data files are available in the "../input/" directory.
from sklearn.impute import SimpleImputer
import os
import gc
import logging
# Any results you write to the current directory are saved as output.

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# global alphabet,max_row_size,nb_classes,nb_epoch,nb_batch_size
nb_classes = 2 # binary classification
nb_epoch = 10 
nb_batch_size = 200
max_row_size = 100 # defined maximum length for smiles string

file_path = 'data/'
# all characters in all SMILES strings
alphabet = pd.read_csv(file_path +'all_alphabet.csv')['alphabet'].tolist()

def smiles_features_data(sml_lst):
    '''
    input : list of smiles strings
    out : dataframe of flattened smiles features
    '''
    una = [] # list of unavailable smls
    feature_df = pd.DataFrame(columns=alphabet*max_row_size, index= sml_lst )
    for i in range(len(sml_lst)):
        sml = sml_lst[i]
        temp = create_smiles_features(sml)
        print(sum(temp))
        if temp is not None:
            feature_df.iloc[i,:]  = temp
        else:
            una.append(sml)
    feature_df_out = feature_df.dropna(axis='index')
    if len(una) > 0:
        print(str(len(una))+' isomeric smiles are unavailable:')
        print(', '.join(una))
    return feature_df_out

# from pathlib import Path

def create_smiles_features(sml):
    '''
    input : smiles strings
    out : flattened smiles features
    '''     
    try:
        sml_data = pcp.get_compounds(sml, 'smiles', as_dataframe=True)
        isomeric_smiles = sml_data.isomeric_smiles.values[0]
        smiles_data_str = list(isomeric_smiles)
        row_index_names = smiles_data_str + ['void_'+ str(s) for s in range(len(smiles_data_str),max_row_size)]
        smiles_data_matrix = pd.DataFrame(columns=alphabet, index= row_index_names )
        # fill the cell by zeros if the length of smiles is less than the defined maximum lenght 
        smiles_data_matrix = smiles_data_matrix.fillna(0) 
        mat_tar = dict()
        for ch in smiles_data_str:
            if ch in alphabet:
                smiles_data_matrix.loc[ch,ch]=1
        # flatten the feature matrix
        mat_tar = smiles_data_matrix.values.flatten()
        return mat_tar
    except Exception as e:
        logger.error(sml+"isomeric smiles is unavailable")
        return None


 # # data preprocessing
def data_prep_1(processed_data,target_index,max_row_size):
    '''
    Prepare the data for the target of interest
    without other tagets as features
    '''
    img_rows, img_cols = max_row_size ,len(alphabet)

    y_index = list(np.arange(0,processed_data.shape[1]-12))+ [processed_data.shape[1]-13+target_index]
    data_ori = processed_data.iloc[:,y_index]
#     print(data_ori.columns)
#     print(data_ori.shape)
    data = data_ori.dropna(axis=0)
#     print(data.shape)
    y_out = data.iloc[:,-1]
    # non-toxic: 0 , toxic: 1
#     x_out = data.iloc[:,:data.shape[1]-1]
    x_out = data.iloc[:,:max_row_size*len(alphabet)]
#     print(x_out.shape)
#     print(y_out.shape)
    return x_out, y_out

def data_prep_2(processed_data,target_index,max_row_size):
    '''
    This function prepares the data for the target of interest 
    which includes the activities of compound on the other tartgets.
    It exports the correct data format of features and labels for training and testing.
    '''
    img_rows, img_cols_pre, img_cols = max_row_size ,len(alphabet),len(alphabet)+11

    y_target_ind =processed_data.shape[1]-13+target_index
    y_target_label = processed_data.columns[y_target_ind]
    y_out = processed_data.iloc[:,y_target_ind]
    y_feature = processed_data.drop(y_target_label, axis=1)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    y_feature_trans = pd.DataFrame(imp.fit_transform(y_feature))
    y_feature_trans.index = processed_data.index
    data_ori =pd.concat([y_feature_trans, y_out], axis=1)
    data = data_ori.dropna(axis=0)
#     print(data.shape)
    y_out = data.iloc[:,-1]
    # non-toxic: 0 , toxic: 1
#     x_out = data.iloc[:,:data.shape[1]-1]
    x_out_a = data.iloc[:,:max_row_size*len(alphabet)].values.reshape(y_out.shape[0], img_rows, img_cols_pre)
    x_out_b = data.iloc[:,-12:-1]
    
    x_out_a_copy = np.empty(shape=(len(x_out_b), img_rows, img_cols))
    for i in np.arange(0,len(x_out_b)):
        a = pd.DataFrame(x_out_a[i])
        b = pd.concat([x_out_b.iloc[i,]]* max_row_size,axis=1).transpose()
        b.index = a.index
        c = pd.concat([a,b],axis=1)
        x_out_a_copy[i]= c.values
    x_out = x_out_a_copy.reshape(y_out.shape[0], 1, img_cols*img_rows)
    return x_out, y_out

def data_prep_weighted(processed_data,target_index,max_row_size,smiles):
    '''
    Prepare the data for the target of interest
    without other tagets as features
    '''
    img_rows, img_cols = max_row_size ,len(alphabet)
    data= processed_data.loc[smiles,:]
    # non-toxic: 0 , toxic: 1
    x_out = data.iloc[:,:max_row_size*len(alphabet)]
    x_out = x_out.values.reshape(len(smiles), img_rows, img_cols,1)
    return x_out

def data_prep_weighted_target(processed_data,target_index,max_row_size,smiles):
    '''
    This function prepares the data for the target of interest 
    which includes the activities of compound on the other tartgets.
    It exports the correct data format of features only for the compounds which the toxicities are previously unknown.
    '''
    img_rows, img_cols_pre, img_cols = max_row_size ,len(alphabet),len(alphabet)+11

    y_target_ind =processed_data.shape[1]-13+target_index
    y_target_label = processed_data.columns[y_target_ind]
    y_out = processed_data.iloc[:,y_target_ind]
    y_feature = processed_data.drop(y_target_label, axis=1)
    
    # fill up the missing data
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    y_feature_trans = pd.DataFrame(imp.fit_transform(y_feature))
    y_feature_trans.index = y_feature.index
    data_ori =pd.concat([y_feature_trans, y_out], axis=1)
    
    # get the rows which have no target activity values
    data = data_ori.loc[smiles,:]
    
    x_out_a = data.iloc[:,:max_row_size*len(alphabet)].values.reshape(data.shape[0], img_rows, img_cols_pre)
    x_out_b = data.iloc[:,-12:-1]
    x_out_a_copy = np.empty(shape=(len(x_out_b), img_rows, img_cols))
    for i in np.arange(0,len(x_out_b)):
        a = pd.DataFrame(x_out_a[i])
        b = pd.concat([x_out_b.iloc[i,]]* max_row_size,axis=1).transpose()
        b.index = a.index
        c = pd.concat([a,b],axis=1)
        x_out_a_copy[i]= c.values
    # reshape and export the features 
    x_out = x_out_a_copy.reshape(data.shape[0], img_rows,img_cols,1)
    return x_out

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Resource: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def plot_trainning(history,mod):
    # plot metrics
    plt.subplot(2,1,1)
    plt.title('Accuracy over epoches - '+ mod)
    plt.plot(history.history['acc'])
    plt.grid(linestyle = '--')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.subplot(2, 1, 2)
    plt.title('losses.binary_crossentropy over epoches - '+ mod)
    plt.plot(history.history['loss'])
    plt.grid(linestyle = '--')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.tight_layout()
    plt.show()

def show_performance(real, pred):
    '''
    show the performance report and the confusion matrix
    '''
    class_names = ['non-toxic', 'toxic']
    report = classification_report(real, pred,target_names=class_names,output_dict=True)
    print(report)
    # cnf_matrix = confusion_matrix(real, pred)
    # np.set_printoptions(precision=2)
    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    # plt.show()
    return report['toxic']['recall']

def performance(real, pred, beta):
    '''
    calculate the recall and f-beta score from the predicted results
    '''
    recall = show_performance(real, pred)
    print('recall - toxic: {0:0.2f}'.format(recall))
    # since the dataset is imblanced, so we use weigthed average F beta score as the performance metrics. 
    # beta is asigned to 0.5 for emphazizing the recall over precision for the positive(toxic class
    fbeta = fbeta_score(real, pred, average='weighted', beta=beta) 
    print('F-beta score: {0:0.2f}'.format(fbeta))
    return recall, fbeta

def build_model(img_rows, img_cols, optimizer= 'adam'):
    '''
    This function creates the CNN model
    '''
    # create model
    model = Sequential()
    # add layer
    model.add(Conv2D(filters = 32, kernel_size=(5, 5),
                     activation='relu',padding = 'SAME',
                     strides=2,
                     input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(filters = 32, kernel_size=(5, 5),
                     activation='relu',padding = 'SAME',
                     strides=2))
    model.add(MaxPool2D(pool_size=(3,3),padding='SAME'))

    model.add(Dropout(0.5))
    
    model.add(Conv2D(filters = 64, strides=2, kernel_size=(3, 3),padding = 'SAME',  activation='relu'))
    model.add(Conv2D(filters = 64, strides=2, kernel_size = (3,3),padding = 'SAME', activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    # avoid overfitting
    model.add(Dropout(0.5))
    
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))   # binary prediciton
    
    # compile model
    model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer= optimizer,
                      metrics=['acc'])
    return model

def toxicity_prediction_weighted_targets(processed_data,target_index):
    '''
    This function mananges the training process for CNNs
    model: class weighted  + toxicity of other targets
    '''
    img_rows, img_cols = max_row_size, len(alphabet)+11
    try:
        print("toxcity prediction for target %d" % target_index)
        print('data preparation ...')
        # dataset for the corresponding taget
        x_f,y_f = data_prep_2(processed_data,target_index,max_row_size)

        # define 5-fold cross validation test harness
        kfold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed)
        cvscores = {'acc':[],'recall':[],'fbeta':[]}
        cvmodel_history = []
        for train, test in kfold.split(x_f,y_f):
            print('new fold ...')
            # data reshape
            train_X = x_f[train,:].reshape(len(y_f.iloc[train]), img_rows, img_cols, 1)
            train_y = y_f.iloc[train]
            val_X = x_f[test,:].reshape(len(y_f.iloc[test]), img_rows, img_cols, 1)
            val_y = y_f.iloc[test]
            
            # the imbalance of the dataset 
            n_non_tox_samples = len(train_y[train_y==0])
            n_tox_samples = len(train_y[train_y==1])

            # class weights
            class_weights={
                1: n_non_tox_samples / n_tox_samples , # toxic, minor class
                0: 1 # non-toxic, major class
            }
            print(class_weights)

            #set early stopping criteria
            pat = 5 #this is the number of epochs with no improvment after which the training will stop
            early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

            #define the model checkpoint callback -> this will keep on saving the model as a physical file
            model_checkpoint = ModelCheckpoint('target'+str(target_index)+'_model_weighted_targets.h5', verbose=1, save_best_only=True)

            # establish the CNN model
            my_model = build_model(img_rows, img_cols)

            print('Training the model ...')
            # train the model with training dataset , 20% data for validation
            history = my_model.fit(train_X, train_y,
                  batch_size= nb_batch_size,
                  epochs=nb_epoch,callbacks=[early_stopping, model_checkpoint], 
                  validation_split = 0.2,
                  class_weight=class_weights)
            # plot_trainning(history,'weighted.targets')
            cvmodel_history.append(history)
            # evaluate the model
            scores = my_model.evaluate(val_X, val_y, verbose=0)
            print("%s: %.2f%%" % (my_model.metrics_names[1], scores[1]*100))
            cvscores['acc'].append(scores[1] * 100)
            val_predictions = my_model.predict(val_X)
            val_pred_round = [round(i[0]) for i in val_predictions] # 0.5 threshold
            [recall,fbeta] = performance(val_y, val_pred_round,1.5)
            cvscores['recall'].append(recall * 100)
            cvscores['fbeta'].append(fbeta * 100)
        return cvscores,cvmodel_history
    except:
        return None

def plot_history_sub(mod,all_cvhistory):
    '''
    This function creates and exports the loss and accuracy values during tranining process.
    '''
    t = 1
    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    for cvhistory_resampled in all_cvhistory:
        plt.subplot(4, 3, t)
        cv = 0
        plt.title('Loss - target'+str(t))
        for history in cvhistory_resampled:
            cv =cv +1
            plt.plot(history.history['loss'],label='train '+str(cv)) 
        plt.grid(linestyle = '--')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim(0,1)
        t = t+1
        plt.legend(loc= 'upper right',fontsize='x-small')
    plt.tight_layout()
    plt.savefig(mod +'_12targets.loss.cv.sub.png')
    
    t = 1
    f = plt.figure()
    f.set_figheight(10)
    f.set_figwidth(10)
    for cvhistory_resampled in all_cvhistory:
        plt.subplot(4, 3, t)
        cv = 0
        plt.title('Accuracy - target'+str(t))
        for history in cvhistory_resampled:
            cv =cv +1
            plt.plot(history.history['acc'],label='train '+str(cv)) 
        plt.grid(linestyle = '--')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.ylim(0,1)
        t = t+1
        plt.legend(loc= 'lower right',fontsize='x-small')
    plt.tight_layout()
    plt.savefig(mod+'_12targets.acc.cv.sub.png')



def barplot_cvscores(mod,all_cvscores,lgd_loc,lgd_ncol):
    '''
    This function creates  and exports the barplot figure of accuracy, f-beta
    and recall(toxic class) on the test dataset.  
    '''
    n_groups = 3

    fig, ax = plt.subplots()

    index = np.arange(n_groups)*6
    bar_width = 0.35
    opacity = 0.7
    
    for i in range(12):
        weighted= pd.DataFrame(all_cvscores[i]).mean().tolist()
        ax.bar(index + bar_width*i, weighted, bar_width,
                    alpha=opacity,label='target'+str(i+1))
        
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(['acc','recall','fbeta'])
    ax.legend(loc = lgd_loc,fontsize='small', ncol = lgd_ncol)
    ax.grid(linestyle = '--')
    plt.ylim(0,100)    

    # plt.show()
    plt.savefig('12targets_'+mod+'_5cv.png')


def toxicity_prediction_weighted(processed_data,target_index):
    '''
    This function mananges the training process for CNNs
    model: class weighted
    '''
    img_rows, img_cols = max_row_size, len(alphabet)
    try:
        print("toxcity prediction for target %d" % target_index)

        print('data preparation ...')
        # dataset for the corresponding taget
        x_f,y_f = data_prep_1(processed_data,target_index,max_row_size)
        
        # define 5-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cvscores = {'acc':[],'recall':[],'fbeta':[]}
        cvmodel_history = []
        for train, test in kfold.split(x_f,y_f):
            # data reshape
            train_X = x_f.iloc[train,:].values.reshape(len(y_f.iloc[train]), img_rows, img_cols, 1)
            train_y = y_f.iloc[train]
            val_X = x_f.iloc[test,:].values.reshape(len(y_f.iloc[test]), img_rows, img_cols, 1)
            val_y = y_f.iloc[test]
            
            # the imbalance of the dataset 
            n_non_tox_samples = len(train_y[train_y==0])
            n_tox_samples = len(train_y[train_y==1])

            # class weights
            class_weights={
                1: n_non_tox_samples / n_tox_samples , # toxic, minor class
                0: 1 # non-toxic, major class
            }
            print(class_weights)

            #set early stopping criteria
            pat = 5 #this is the number of epochs with no improvment after which the training will stop
            early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

            #define the model checkpoint callback -> this will keep on saving the model as a physical file
            model_checkpoint = ModelCheckpoint('target'+str(target_index)+'_model_weighted.h5', verbose=1, save_best_only=True)

            # establish the CNN model
            my_model = build_model(img_rows, img_cols)

            print('Training the model ...')
            # train the model with training dataset , 20% data for validation
            history = my_model.fit(train_X, train_y,
                  batch_size= nb_batch_size,
                  epochs=nb_epoch,callbacks=[early_stopping, model_checkpoint], 
                  validation_split = 0.2,
                  class_weight=class_weights)
            # plot_trainning(history,'weighted')
            cvmodel_history.append(history)
            # evaluate the model
            scores = my_model.evaluate(val_X, val_y, verbose=0)
            print("%s: %.2f%%" % (my_model.metrics_names[1], scores[1]*100))
            cvscores['acc'].append(scores[1] * 100)
            val_predictions = my_model.predict(val_X)
            val_pred_round = [round(i[0]) for i in val_predictions] # 0.5 threshold
            [recall,fbeta] = performance(val_y, val_pred_round,1.5)
            cvscores['recall'].append(recall * 100)
            cvscores['fbeta'].append(fbeta * 100)
        return cvscores,cvmodel_history
    except:
        return None

def toxicity_prediction_resampled(processed_data,target_index):
    '''
    This function mananges the training process for CNNs
    model: fit to oversampled the minority class
    '''
    img_rows, img_cols = max_row_size, len(alphabet)
    try:
        print("toxcity prediction for target %d" % target_index)

        print('data preparation ...')
        # dataset for the corresponding taget
        x_f,y_f = data_prep_1(processed_data,target_index,max_row_size)

        
        # define 5-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cvscores = {'acc':[],'recall':[],'fbeta':[]}
        cvmodel_history = []
        for train, test in kfold.split(x_f,y_f):
            # Oversampling the minority class
            sm = SMOTE(random_state=seed,sampling_strategy='minority')
            train_X_res, train_y_res = sm.fit_resample(x_f.iloc[train,:], y_f.iloc[train])

            # data reshape
            train_X_res = train_X_res.reshape(len(train_y_res), img_rows, img_cols, 1)
            val_X = x_f.iloc[test,:].values.reshape(len(y_f.iloc[test]), img_rows, img_cols, 1)
            val_y = y_f.iloc[test]

            #set early stopping criteria
            pat = 5 #this is the number of epochs with no improvment after which the training will stop
            early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

            #define the model checkpoint callback -> this will keep on saving the model as a physical file
            model_checkpoint = ModelCheckpoint('target'+str(target_index)+'_model_resampled.h5', verbose=1, save_best_only=True)

            # establish the CNN model
            my_model = build_model(img_rows, img_cols)

            print('Training the model ...')
            # train the model with training dataset , 20% data for validation
            history = my_model.fit(train_X_res, train_y_res,
                  batch_size= nb_batch_size,
                  epochs=nb_epoch,callbacks=[early_stopping, model_checkpoint], 
                  validation_split = 0.2)
    #               class_weight=class_weights)
            # plot_trainning(history,'resampled')
            cvmodel_history.append(history)
            # evaluate the model
            scores = my_model.evaluate(val_X, val_y, verbose=0)
            print("%s: %.2f%%" % (my_model.metrics_names[1], scores[1]*100))
            cvscores['acc'].append(scores[1] * 100)
            val_predictions = my_model.predict(val_X)
            val_pred_round = [round(i[0]) for i in val_predictions] # 0.5 threshold
            [recall,fbeta] = performance(val_y, val_pred_round,1.5)
            cvscores['recall'].append(recall * 100)
            cvscores['fbeta'].append(fbeta * 100)
        return cvscores,cvmodel_history
    except:
        return None