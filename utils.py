import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sma
from os.path import join

def load_data(input_dir):
    testdata = pickle.load(open(join(input_dir, 'img_features.p'),'rb'))
    clinical = pd.read_csv(join(input_dir, 'clinical.csv'))
    model = tf.keras.models.load_model(join(input_dir, 'pretrained/image.h5'))
    
    return testdata, clinical, model

def predict_image_features(testdata, clinical, model):
    y_pre = []
    for id_p in clinical.case_submitter_id:
        T_test = testdata[id_p].astype(np.float32)
        probs = model.predict(T_test,verbose=0)
        tprobs = probs[:, 1]
        cnt = sum(tprobs > 0.5)
        y_pre.append(cnt / len(tprobs))
    
    return y_pre

def calculate_survival(y_pre, clinical):
    df = pd.DataFrame({
        'id': clinical.case_submitter_id,
        'pre': y_pre
    })
    
    YSC = np.array(clinical[['vital_status','os']])
    T1 = T2 = 0.5
    Rscore = df['pre']

    sf1 = sma.SurvfuncRight(list(YSC[Rscore < T1, 1]), list(YSC[Rscore < T1, 0]))
    sf2 = sma.SurvfuncRight(YSC[Rscore > T2, 1], YSC[Rscore > T2, 0])

    return sf1, sf2

def survival_function_value(i, sp, st):
    """
    Helper function to calculate survival function value
    """
    if len(st) < 1:
        return 1
    if i > np.max(st):
        return sp[-1]
    if i < np.min(st):
        return 1
    sind = np.argmax(st > i) - 1
    return sp[sind]

