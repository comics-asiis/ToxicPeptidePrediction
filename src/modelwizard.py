import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import accessdata
import extractfeature

def Load4modelsAndPredict(input_fastapath, ul_seqlen=50, rnn_cutoff=0.5, output_dir=None):
    pred_seqDi, notproc_seqDi = accessdata.ReadFasta(input_fastapath, ul_seqlen=ul_seqlen)
    model_svc = pickle.load(open('model_svc.pickle', 'rb'))
    scaler_svc = pickle.load(open('scaler_svc.pkl', 'rb'))
    model_rf = pickle.load(open('model_rf.pickle', 'rb'))
    scaler_rf = pickle.load(open('scaler_rf.pkl', 'rb'))
    model_gbc = pickle.load(open('model_gbc.pickle', 'rb'))
    scaler_gbc = pickle.load(open('scaler_gbc.pkl', 'rb'))
    pred_label_svc = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_svc,
                                            scaler=scaler_svc, max_seq_len=ul_seqlen)
    pred_label_rf = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_rf,
                                           scaler=scaler_rf, max_seq_len=ul_seqlen)
    pred_label_gbc = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_gbc,
                                            scaler=scaler_gbc, max_seq_len=ul_seqlen)
    df_pred_result = extractfeature.GenerateInitialSequenceDataframe(pred_seqDi, islabeled=False)
    df_pred_result['SVM prediction'] = pred_label_svc
    df_pred_result['RF prediction'] = pred_label_rf
    df_pred_result['GBM prediction'] = pred_label_gbc
    pred_label_rnn = PredictSeqUsingRNNmodel(pred_seqDi, max_seq_len=ul_seqlen, cutoff=rnn_cutoff)
    df_pred_result['RNN prediction'] = pred_label_rnn

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d@%H%M%S")
    result_csv_name = 'prediction_result_' + current_time + '.csv'
    result_csv_path = result_csv_name if output_dir is None else output_dir + '/' + result_csv_name
    accessdata.WriteDataframe2Csv(df_pred_result, result_csv_path)
    print('Prediction results have been generated.')

    if len(notproc_seqDi) > 0:
        print('There are sequences not processed for length larger than', ul_seqlen, ':')
        for header in notproc_seqDi:
            print('>', header, sep='')
            print(notproc_seqDi[header])

    return result_csv_name


def LoadRNNmodelAndPredict(fastapath, ul_seqlen=50):
    pred_seqDi, notproc_seqDi = accessdata.ReadFasta(fastapath)
    pred_label_rnn = PredictSeqUsingRNNmodel(pred_seqDi, max_seq_len=ul_seqlen)
    input_seq_di = pred_seqDi
    df_pred_result = extractfeature.GenerateInitialSequenceDataframe(input_seq_di, islabeled=False)
    df_pred_result['RNN prediction'] = pred_label_rnn

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d@%H%M%S")
    result_csv_name = 'prediction_result_RNN_' + current_time + '.csv'
    accessdata.WriteDataframe2Csv(df_pred_result, result_csv_name)
    print('Have generate prediction results from the RNN model')


def LoadMLmodelAndPredict(fastapath, ul_seqlen=50):
    pred_seqDi, notproc_seqDi = accessdata.ReadFasta(fastapath)
    model_svc = pickle.load(open('model_svc.pickle', 'rb'))
    scaler_svc = pickle.load(open('scaler_svc.pkl', 'rb'))
    model_rf = pickle.load(open('model_rf.pickle', 'rb'))
    scaler_rf = pickle.load(open('scaler_rf.pkl', 'rb'))
    model_gbc = pickle.load(open('model_gbc.pickle', 'rb'))
    scaler_gbc = pickle.load(open('scaler_gbc.pkl', 'rb'))

    pred_label_svc = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_svc,
                                            scaler=scaler_svc, max_seq_len=ul_seqlen)
    pred_label_rf = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_rf,
                                            scaler=scaler_rf, max_seq_len=ul_seqlen)
    pred_label_gbc = PredictSeqUsingMLmodel(pred_seqDi, None, encode_profile=True, profile_type=0, model=model_gbc,
                                            scaler=scaler_gbc, max_seq_len=ul_seqlen)
    input_seq_di = pred_seqDi
    df_pred_result = extractfeature.GenerateInitialSequenceDataframe(input_seq_di, islabeled=False)
    df_pred_result['SVM prediction'] = pred_label_svc
    df_pred_result['RF prediction'] = pred_label_rf
    df_pred_result['GBM prediction'] = pred_label_gbc

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d@%H%M%S")
    result_csv_name = 'prediction_result_ML_' + current_time + '.csv'
    accessdata.WriteDataframe2Csv(df_pred_result, result_csv_name)
    print('Have generate prediction results from 3 ML models')


def PredictSeqUsingRNNmodel(pred_seqDi, given_features=None, profile_type=0, max_seq_len=50, cutoff=0.5):
    feat_pcp13 = ['pKa1', 'pKa2', 'Vol', 'VSC', 'SASA', 'pI', 'Chg', 'NCIS', 'Hyd1', 'Hydro', 'FEtmh', 'Pol1', 'Pol2']
    considered_features = feat_pcp13 if given_features is None else given_features

    initial_df_sequence_pred = extractfeature.GenerateInitialSequenceDataframe(pred_seqDi, islabeled=False)
    x_pred = None
    if profile_type == 0:
        y_pred, x_pred = extractfeature.ConstructSequenceProfile_GivenPropertyFeatures(initial_df_sequence_pred,
                                                                                       considered_features,
                                                                                       padding_length=max_seq_len,
                                                                                       reshape=True,
                                                                                       with_label=False)

    model = RecurrentModel_OneGruLayer(x_pred,num_unit=64)
    with tf.device('/cpu:0'):
        model.load_weights('model_rnn.h5')
        pred_prob = model.predict(x_pred)
        pred_label = np.where(pred_prob > cutoff, 1, 0).flatten()
    return pred_label


def PredictSeqUsingMLmodel(pred_seqDi, given_features, encode_profile, profile_type=0, model=None, scaler=None, max_seq_len=50):
    feat_adnc_pcp13 = ['AAC', 'DPC', 'NBP', 'CBP', 'pKa1', 'pKa2', 'Vol', 'VSC', 'SASA', 'pI', 'Chg', 'NCIS', 'Hyd1',
                       'Hydro', 'FEtmh', 'Pol1', 'Pol2']

    considered_features = feat_adnc_pcp13 if given_features is None else given_features
    reshape = False
    x_pred = None

    if encode_profile is True:
        initial_df_sequence_pred = extractfeature.GenerateInitialSequenceDataframe(pred_seqDi, islabeled=False)
        if profile_type == 0:
            y_pred, x_pred = extractfeature.ConstructSequenceProfile_GivenPropertyFeatures(initial_df_sequence_pred,
                                                                                             considered_features,
                                                                                             padding_length=max_seq_len,
                                                                                             reshape=reshape, with_label=False)

        if ('AAC' or 'DPC' or 'NBP' or 'CBP') in considered_features and reshape is False: 
            composition_based_feature_li = []
            if 'AAC' in considered_features:
                composition_based_feature_li.append('AAC')
            if 'DPC' in considered_features:
                composition_based_feature_li.append('DPC')
            if 'NBP' in considered_features:
                composition_based_feature_li.append('NBP')
            if 'CBP' in considered_features:
                composition_based_feature_li.append('CBP')
            predDf = extractfeature.GenerateFeatureTableGivenSeqDiAndFeatureLi(pred_seqDi, composition_based_feature_li, islabeled=False)
            x_pred_cbf = predDf[predDf.columns[~predDf.columns.isin(['Name', 'Label', 'Sequence', 'Length'])]] # cbf means combined feature

            x_pred_cbf = scaler.transform(x_pred_cbf)
            x_pred = np.concatenate((x_pred, x_pred_cbf), axis=1)
            # print('concatenating composition-based features')
    else:
        predDf = extractfeature.GenerateFeatureTableGivenSeqDiAndFeatureLi(pred_seqDi, considered_features, islabeled=False)
        x_pred = predDf[predDf.columns[~predDf.columns.isin(['Name', 'Label', 'Sequence', 'Length'])]]
        x_pred = scaler.transform(x_pred)

    pred_label = model.predict(x_pred)
    return pred_label


def RecurrentModel_OneGruLayer(input_feature, num_unit=64):
    sequence_length = input_feature.shape[1]
    dimension_eachaa = input_feature.shape[2]
    model_gru = keras.Sequential()
    model_gru.add(layers.GRU(num_unit, input_shape=(sequence_length, dimension_eachaa)))
    model_gru.add(layers.Dense(1, activation='sigmoid'))
    return model_gru
