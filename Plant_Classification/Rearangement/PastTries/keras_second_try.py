import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow.keras.backend as K

import cv2
import os.path
import numpy as np

import sys

def create_models(h, w):

    ######################################################################
    # create a Given_Model whose inputs are the image and outputs are
    # probability of plant

    # input is the RGB image, which is h-by-w-by-3
    img_in = keras.Input(shape=(h, w, 3), name='img_in') #img_in = x and replace.

    x=img_in
    x = layers.SeparableConv2D(4, 3, padding="same",activation='relu')(x)
    x = layers.SeparableConv2D(8, 3, padding="same", activation='relu')(x)


    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(4, 3, padding="same", activation='relu')(x)
    x = layers.SeparableConv2D(8, 3, padding="same", activation='relu')(x)
    #x = layers.SeparableConv2D(16, 3, padding="same", activation='relu')(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(8, 3, padding="same", activation='relu')(x)
    x = layers.Conv2DTranspose(4, 3, padding="same",activation='relu')(x)





    # sigmoid is the appropriate activation for logistic regression
    prob_plant = keras.layers.Dense(1, activation='sigmoid', name='my_dense')(x)

    # instead of (batch_size, h, w, 1) do (batch_size, h, w)
    prob_plant = keras.layers.Reshape((h, w))(prob_plant) # no need for 1?

    # here's our classification-only Given_Model
    classifier_only = keras.Model(img_in, prob_plant)

    ######################################################################
    # create another Given_Model whose inputs are the image, weights, and
    # labels that we can add custom loss and metrics to

    # additional input for per-pixel weights (use just for training)
    weights_in = keras.Input(shape=(h, w), name='weights_in')

    # additional input for true labels (use just for training)
    true_labels = keras.Input(shape=(h, w), name='true_labels')

    # compute binary cross-entropy per-pixel, used for loss
    bce = tf.keras.metrics.binary_crossentropy(true_labels, prob_plant, axis=())

    # to get loss, we weight this cross-entropy loss per-pixel and sum it over the image
    loss = tf.reduce_sum(tf.multiply(bce, weights_in), axis=(1, 2), name='loss')

    # we predict plant whenever p(plant) > 0.5
    pred_labels = tf.cast(tf.greater(prob_plant, 0.5), tf.float32, name='pred_labels')

    # per-pixel correct/incorrect as float32
    is_correct = tf.cast(tf.equal(pred_labels, true_labels), tf.float32, name='is_correct')

    # weighted accuracy: weighted sum of is_correct over spatial dimensions
    weighted_accuracy = tf.reduce_sum(tf.multiply(is_correct, weights_in), axis=(1, 2), name='weighted_accuracy')

    # our full Given_Model has 3 inputs: image, weights, labels
    model = keras.Model([img_in, weights_in, true_labels], [])

    # add the custom per-pixel loss function we created
    model.add_loss(loss)

    # add metrics to compute weighted and unweighted accuracy during training
    model.add_metric(weighted_accuracy, 'weighted_accuracy')

    # compile the Given_Model
    model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=None)

    # print it out
    model.summary()

    return classifier_only, model


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

    sample_weight = np.zeros((h, w), dtype=np.float32)
    sample_weight[pos_mask] = pos_weight
    sample_weight[neg_mask] = neg_weight

    any_mask = pos_mask | neg_mask

    #any_weight = 1.0 / any_mask.sum()
    #sample_weight[any_mask] = any_weight

    print('should be 1:', sample_weight.sum())

    ######################################################################
    # create models

    classifier_only, full_model = create_models(h, w)
    
    
    # fit to data
    full_model.fit(x=(img[None, :], sample_weight[None, :], labels[None, :]),
                   epochs=150)

    weights, bias = full_model.get_layer('my_dense').get_weights()

    print('weights:', weights)
    print('bias:', bias)

    pred_probs = classifier_only.predict(img[None, :])
    pred_probs = pred_probs[0] # get rid of 'batch' dimension
    
    pred_labels = np.round(pred_probs) # 1's and 0's


    correct = (labels[any_mask] == pred_labels[any_mask])

    weighted_accuracy = (correct * sample_weight[any_mask]).sum()
    unweighted_accuracy = correct.mean()

    print('weighted accuracy on just masked pixels:', weighted_accuracy)
    print('unweighted accuracy on just masked pixels:', unweighted_accuracy)

   # newimg = cv2.imread("test_11000_10000.tif")
   #evaluate full_model.evaluate()

   #1 Eval on second image
   #2 convert to point/region
   #3 Evaluate end to end accuracy

   #Save/Loading Keras Model or Weights
   #build funcs to calculate
   #Make py module



if __name__ == '__main__':
    main()

#padding valid
#batchnorm pad
#blur 4 8 16
#up 16 8 4
