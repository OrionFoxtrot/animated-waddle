import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def Read_Image(imgfile):
    (base, ext) = os.path.splitext(imgfile)
    print('base:', base, 'ext:', ext)
    txtfile = base + ".txt"

    img = cv2.imread(imgfile)
    img = img.astype(np.float32)
    img -= img.mean(axis=(0, 1))
    img /= img.std(axis=(0, 1))

    data = np.genfromtxt(txtfile, delimiter=' ', dtype=float, unpack=True)
    x, y = data

    return img, x, y





def Generate_Masks(x, y):
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
    return pos_mask, neg_mask


def Generate_Weights(img, pos_mask, neg_mask):
    h, w = img.shape[:2]

    # labels is 0.5 where neither mask hits
    labels = np.full((h, w), 0.5, dtype=np.float32)
    labels[pos_mask] = 1.0
    labels[neg_mask] = 0.0  # not -1 because sigmoid (logistic regression)

    neg_weight = 0.5 / neg_mask.sum()
    pos_weight = 0.5 / pos_mask.sum()

    sample_weight = np.zeros((h, w), dtype=np.float32)
    sample_weight[pos_mask] = pos_weight
    sample_weight[neg_mask] = neg_weight

    any_mask = pos_mask | neg_mask

    return sample_weight, any_mask, labels, pos_weight,neg_weight


def convert(probability_array, imgfile,imgfile1, type=0, lsq = 0, ):
    probability_array = probability_array[0, :, :]

    img_gray = (probability_array * 255).astype(np.uint8)

    if (type == 1):#MINMAX
        centers = []
        mask = np.full(probability_array.shape, 255, dtype=np.uint8)
        while (True):
            _, Max, _, MaxLoc = cv2.minMaxLoc(probability_array, mask)
            centers.append(MaxLoc)
            # print(Max,",",MaxLoc)
            if (Max < 0.5):
                break
            else:
                cv2.circle(mask, MaxLoc, 15, (0, 0, 0), -1)
        img_thresh = ~mask
        centers = np.array(centers)

    if(type == 0):#if type = 0 basically
        img_thresh = np.where(probability_array > 0.5, np.uint8(255), np.uint8(0))

        if(lsq == 1):
            img_thresh, img_gray = load_least_squares(imgfile,imgfile1)

        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for (i, contour) in enumerate(contours):

            m = cv2.moments(contour)

            area = m['m00']
            if (area == 0): area = 1
            cx = m['m10'] / area
            cy = m['m01'] / area
            centers.append((cx, cy))

            # print(f'  - contour {i} has {len(contour)} points, area={area}, center at ({cx}, {cy})')
        centers = np.array(centers, dtype=np.float32)

    np.savetxt("CentersList.txt", centers, "%.2f")
    return (img_gray, img_thresh, centers)


def create_models(h, w):
    ######################################################################
    # create a Given_Model whose inputs are the image and outputs are
    # probability of plant

    # input is the RGB image, which is h-by-w-by-3
    img_in = keras.Input(shape=(h, w, 3), name='img_in')  # img_in = x and replace.

    x = img_in
    x = layers.SeparableConv2D(4, 3, padding="same", activation='relu')(x)
    x = layers.SeparableConv2D(8, 3, padding="same", activation='relu')(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = layers.SeparableConv2D(4, 3, padding="same", activation='relu')(x)
    x = layers.SeparableConv2D(8, 3, padding="same", activation='relu')(x)
    # x = layers.SeparableConv2D(16, 3, padding="same", activation='relu')(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(8, 3, padding="same", activation='relu')(x)
    x = layers.Conv2DTranspose(4, 3, padding="same", activation='relu')(x)

    # sigmoid is the appropriate activation for logistic regression
    prob_plant = keras.layers.Dense(1, activation='sigmoid', name='my_dense')(x)

    # instead of (batch_size, h, w, 1) do (batch_size, h, w)
    prob_plant = keras.layers.Reshape((h, w))(prob_plant)  # no need for 1?

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


def Average(lst):
    return sum(lst) / len(lst)


def optimze_point_dist(points1, points2,img):
    points1 = points1
    points2 = points2

    C = cdist(points1, points2)

    assignment1, assigment2 = linear_sum_assignment(C)

    fig, ax = plt.subplots(1, 1)
    ax.plot(points1[:, 0], points1[:, 1], 'bo', markersize=2)
    ax.plot(points2[:, 0], points2[:, 1], 'rs', markersize=2)
    count = 0

    for (idx1, idx2) in zip(assignment1, assigment2):
        dist = C[idx1, idx2]
        if dist < 5:
            p1 = points1[idx1]
            p2 = points2[idx2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
            count += 1

    plt.imshow(img[:,:,::-1],interpolation='none',filternorm=False)

    plt.savefig("Test.pdf")
    accuracy = count / (len(points1) + len(points2) - count)
   # print("max dist", max(dists), "min", min(dists), "average", Average(dists))
    print("Associated Points", count, "out of a total", len(points2), "detected points and", len(points1),
          "truth points with accuracy",accuracy)

    # plt.show()

def load_least_squares(imgfile,imgfile1):
    print("Via LSQ")
    img,x,y = Read_Image(imgfile)
    img1,x1,y1 = Read_Image(imgfile1)
    pos_mask,neg_mask = Generate_Masks(x,y)
    pos_mask1,neg_mask1 = Generate_Masks(x1,y1)

    blursizes = [5, 11, 21]
    my_list = [img]
    my_second_list = [img1]

    for size in blursizes:
        current_blur = cv2.blur(img, (size, size))
        current_blur1 = cv2.blur(img1,(size,size))
        my_list.append(current_blur)
        my_second_list.append(current_blur1)

    megaimg = np.dstack(my_list)
    pos_pixels = megaimg[pos_mask]
    neg_pixels = megaimg[neg_mask]

    megaimg1 = np.dstack(my_second_list)
    pos_pixels1 = megaimg1[pos_mask1]
    neg_pixels1 = megaimg1[neg_mask1]


    A = np.vstack((pos_pixels, neg_pixels))
    A1 = np.vstack((pos_pixels1,neg_pixels1))

    A_Len = len(A)
    A1_Len = len(A1)

    A = np.append(A, np.ones([A_Len, 1]), 1)
    A1 = np.append(A1,np.ones([A1_Len,1]),1)

    Y = np.ones([A_Len])
    Y1 = np.ones([A1_Len])

    Y[len(pos_pixels):] = -1
    Y1[len(pos_pixels1):]=-1

    _,_,_,WiP,WiN = Generate_Weights(img,pos_mask,neg_mask)
    _,_,_,WiP1,WiN1 = Generate_Weights(img1,pos_mask1,neg_mask1)

    w = np.zeros(len(A))
    w[0:len(pos_pixels)] = WiP
    w[len(pos_pixels):] = WiN

    w1 = np.zeros(len(A1))
    w1[0:len(pos_pixels1)]=WiP1
    w[len(pos_pixels1):] = WiN1

    A *= np.sqrt(w[:, None])  # Generate Augmented Matrix's
    Y *= np.sqrt(w)

    A1 *= np.sqrt(w1[:,None])
    Y1 *= np.sqrt(w1)

    print("YS",Y.shape)
    print("Y1S",Y1.shape)

    A = np.vstack([A,A1])
    Y = np.hstack([Y,Y1])
    print("NYS",Y.shape)

    lamda = 1e-5  # Regularization constant
    regularization = lamda * np.identity((A.T @ A).shape[0])
    inverse = np.linalg.inv(A.T @ A + regularization)
    Sol = inverse @ (A.T @ Y)
    np.savetxt("LeastSquaresSol.txt", Sol)

    constants = (len(blursizes) + 1) * 3
    img_gray = (megaimg * Sol[:constants]).sum(axis=-1) + Sol[constants]  # 3 becomes n constants, or chanels.
    img_thresh = np.where(img_gray > 0, np.uint8(255), np.uint8(0))

    imin = img_gray.min()
    imax = img_gray.max()
    img_gray = (img_gray - imin) / (imax - imin)
    return img_thresh,img_gray


# Displays images of the tuple "imgs"
def display_results(imgs):
    print("Results:")
    i = 0
    for img in imgs:
        cv2.imshow(str(i), img)
        i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def main():
    print("Mp")

if __name__ == "__main__":
    main()
