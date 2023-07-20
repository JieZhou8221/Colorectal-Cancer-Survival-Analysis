import tensorflow as tf
import os
import pickle
import random
import numpy as np
import cv2

def process_images(imgall, imgpath, savepath):
    random.shuffle(imgall)

    # Initializing model only once rather than in the loop
    mpln = 512
    base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet',
                                                                 input_shape=(mpln, mpln, 3), classes=2)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

    for pa in imgall:
        if not os.path.exists(savepath + pa):
            coor0 = pickle.load(open(imgpath + pa, 'rb'))
            process_image(coor0, model, savepath, pa)

def process_image(coor0, model, savepath, pa):
    tiles = coor0['tiles']
    coor = coor0['coor']
    tnum = len(tiles)
    ims = np.zeros((tnum, 512, 512, 3), dtype=np.uint8)

    for i in range(tnum):
        ctile = cv2.resize(np.uint8(tiles[i]), (512, 512))
        ims[i] = ctile

    ci1 = model.predict(tf.keras.applications.inception_v3.preprocess_input(ims))
    data = {'incep_raw': ci1, 'coor': coor, 'original_dim': coor0['original_dim']}

    pickle.dump(data, open(savepath + pa, "wb"))



