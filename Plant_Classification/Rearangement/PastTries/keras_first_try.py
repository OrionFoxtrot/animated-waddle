import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow.keras.backend as K

#intersection over union - For metric not pixels
import cv2
import os.path
import numpy as np

def main():


    imgfile = "test_1000_6000.tif"

    (base, ext) = os.path.splitext(imgfile)
    print('base:', base, 'ext:', ext)
    txtfile = base+".txt"

    # -----S:READ IMAGE-----#
    img = cv2.imread(imgfile)
    img = img.astype(np.float32)
    img -= img.mean(axis=(0, 1))
    img /= img.std(axis=(0, 1))
    # ------E:READ IMAGE-----#

    # -----S:POSITIVE LOCATION-----#
    data = np.genfromtxt(txtfile, delimiter=' ', dtype=float, unpack=True)
    x, y = data
    # -----E:POSITIVE LOCATION-----#

    # -----S:GENERATE MASKS-----#
    pos_mask = np.zeros((1000, 1000), np.uint8)  # 1000x1000 black / 0
    neg_mask = np.full((1000, 1000), 255, np.uint8)  # 1000x1000 white / 255

    x, y = np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)
    for index in range(len(x)):  # Drawing Masks
        pos_mask = cv2.circle(img=pos_mask, center=(x[index], y[index])
                              , radius=3, color=(255, 255, 255), thickness=-1)
        neg_mask = cv2.circle(img=neg_mask, center=(x[index], y[index])
                              , radius=7, color=(0, 0, 0), thickness=-1)
    pos_mask = pos_mask.astype(bool)  # Making Masks
    neg_mask = neg_mask.astype(bool)

    ######################################################################

    h, w = img.shape[:2]


    # labels is 0.5 where neither mask hits
    labels = np.full((h, w), 0.5, dtype=np.float32)
    labels[pos_mask] = 1.0
    labels[neg_mask] = 0.0 # not -1 because sigmoid (logistic regression)

    neg_weight = 0.5 / neg_mask.sum()
    pos_weight = 0.5 / pos_mask.sum()

    sample_weight = np.zeros((h, w, 1), dtype=np.float32)
    sample_weight[pos_mask] = pos_weight
    sample_weight[neg_mask] = neg_weight


    ######################################################################

    # input is the RGB image, which is h-by-w-by-3
    inputs = keras.Input(shape=(h, w, 3))

    # sigmoid is the appropriate activation for logistic regression
    outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)

    model = keras.Model(inputs, outputs)

    # appropriate loss for logistic regression
    bce = keras.losses.BinaryCrossentropy()

    model.compile(optimizer=keras.optimizers.Adam(1e-1),
                  loss=bce,
                  metrics=['accuracy'], 
                  sample_weight_mode='temporal')

    # prints the Given_Model structure
    model.summary()

    # feed in the image and labels 
    model.fit(img[None, :], labels[None, :], epochs=25)

    weights, bias = model.layers[1].get_weights()

    print('weights:', weights)
    print('bias:', bias)

    pred_probs = model.predict(img[None, :])

    pred_probs = pred_probs[0, :, :, 0] # get rid of 'batch' dimension
    
    pred_labels = np.round(pred_probs) # 1's and 0's

    #print(labels.shape, pred_labels.shape)

    correct = (labels[pos_mask | neg_mask] == pred_labels[pos_mask | neg_mask])

    unweighted_accuracy = correct.mean()

    print('unweighted accuracy:', unweighted_accuracy)
                    


if __name__ == '__main__':
    main()


