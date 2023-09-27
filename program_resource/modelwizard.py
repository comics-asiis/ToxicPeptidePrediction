import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import accessdata
import extractfeature

""" The main body of the process for toxicity prediction """
def load_4models_predict(input_fastapath, ll_seqlen=10, ul_seqlen=50, given_features=None, output_dir=None, max_seq_num=10000):
    pred_seq_di, notproc_seq_di = accessdata.readfasta(input_fastapath, ul_seqlen=ul_seqlen, ll_seqlen=ll_seqlen)
    if len(pred_seq_di) > max_seq_num:
        print('Not allowed to process more than ' + str(max_seq_num) + 'sequences')
        return "No result"
    
    # Load 4 ML models: LR, SVM, RF, & XGBoost from pickle files (4 pairs of models and scalers)
    current_dir = os.getcwd()
    model_lr = pickle.load(open(current_dir + '/model_lr.pkl', 'rb'))
    scaler_lr = pickle.load(open(current_dir + '/scaler_lr.pkl', 'rb'))

    model_svm = pickle.load(open(current_dir + '/model_svm.pkl', 'rb'))
    scaler_svm = pickle.load(open(current_dir + '/scaler_svm.pkl', 'rb'))

    model_rf = pickle.load(open(current_dir + '/model_rf.pkl', 'rb'))
    scaler_rf = pickle.load(open(current_dir + '/scaler_rf.pkl', 'rb'))

    model_xgb = pickle.load(open(current_dir + '/model_xgboost.pkl', 'rb'))
    scaler_xgb = pickle.load(open(current_dir + '/scaler_xgboost.pkl', 'rb'))
   
    # Encode input seqeucnes as features and then predict their toxicity
    feat_adnc_pcp13 = ['AAC', 'DPC', 'NBP', 'CBP', 'pKa1', 'pKa2', 'Vol', 'VSC', 'SASA', 'pI', 'Chg', 'NCIS', 'Hyd1',
                       'Hyd2', 'Hydro', 'FEtmh', 'Pol2']
    considered_features = feat_adnc_pcp13 if given_features is None else given_features

    pred_lr = EncodeTestDataAndPredict(pred_seq_di, considered_features, model_lr, scaler_lr)
    pred_svm = EncodeTestDataAndPredict(pred_seq_di, considered_features, model_svm, scaler_svm)
    pred_rf = EncodeTestDataAndPredict(pred_seq_di, considered_features, model_rf, scaler_rf)
    pred_xgb = EncodeTestDataAndPredict(pred_seq_di, considered_features, model_xgb, scaler_xgb)

    # Create the dataframe for the prediction results
    df_pred_result = extractfeature.GenerateInitialSequenceDataframe(pred_seq_di, islabeled=False)
    df_pred_result['LR prediction'] = pred_lr
    df_pred_result['SVM prediction'] = pred_svm
    df_pred_result['RF prediction'] = pred_rf
    df_pred_result['XGBoost prediction'] = pred_xgb

    # Export the prediction results to a CSV file
    result_csv_name = input_fastapath + '.csv'
    result_csv_path = result_csv_name if output_dir is None else output_dir + '/' + result_csv_name
    df_pred_result.to_csv(result_csv_path)
    return result_csv_name

def EncodeTestDataAndPredict(seq_di, considered_features, given_model, given_scaler):
    # Transform the input sequences into feature table
    df_data = extractfeature.GenerateFeatureTableGivenSeqDiAndFeatureLi(seq_di, considered_features, islabeled=False) 
    x_data = df_data[df_data.columns[~df_data.columns.isin(['Name', 'TrueLabel', 'Sequence', 'Length'])]]
    
    # Features belonging to physicochemical properties need to be normalized using given_scaler
    pcp13_feats = ['pKa1', 'pKa2', 'Vol', 'VSC', 'SA', 'pI', 'Chg', 'NCIS', 'Hyd1', 'Hyd2', 'Hydro', 'FEtmh', 'Pol2']
    feats_needstandardize = [x for x in pcp13_feats if x in x_data.columns]  
    df_needstddz = x_data[feats_needstandardize]
    df_stddzed = pd.DataFrame(given_scaler.transform(df_needstddz), columns=feats_needstandardize)
    x_data_todrop = x_data.copy()
    x_data_todrop.drop(labels=feats_needstandardize, axis="columns", inplace=True)
    x_data = pd.concat([df_stddzed, x_data_todrop], axis=1, join='inner')
    
    # Predict with the given_model
    pred_label = given_model.predict(x_data)

    return pred_label
