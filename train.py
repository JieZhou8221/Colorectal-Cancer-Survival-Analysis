import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_dataset(filepath):
    data = pickle.load(open(filepath, 'rb'))
    data = data.dropna()
    return data

def training_model(x_train, y_train, fdim, dp, numclasses, ep, bs):
    w = np.sum(y_train)/len(y_train)
    print(w,len(x_train))

    xin = tf.keras.layers.Input(shape=(fdim))
    x=tf.keras.layers.Dense(1024,activation='relu',kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001), kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))(xin)
    x = tf.keras.layers.Dropout(dp)(x)
    xout = tf.keras.layers.Dense(numclasses, activation=tf.nn.softmax,
                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))(x)
    model = tf.keras.models.Model(inputs=xin, outputs=xout)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    sample_weight = np.ones(shape=(len(x_train),))
    sample_weight[y_train == 0] = w
    model.fit(x_train, to_categorical(y_train), epochs=ep, batch_size=bs, sample_weight=sample_weight, shuffle=True,verbose=1)
    return model

def save_predictions(df, id_list, model, typechoose, path, filename):
    preds = []
    for id in id_list:
        x_test = df.loc[df.ID==id,typechoose]
        x_test = np.array(x_test).astype(np.float32)
        probs = model.predict(x_test)[:, 1]
        preds.append(np.sum(probs)/len(probs))

    df_test = df.loc[df.ID.isin(id_list)].reset_index(drop=True)
    df_test['pre'] = pd.DataFrame({'ID':id_list, 'pre':preds}).set_index('ID')['pre']
    df_test.to_csv(path + filename + '.csv',index=False)
