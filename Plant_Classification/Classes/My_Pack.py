import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras import layers


class labeled_data:
    def __init__(self, filename, img, points, pos_mask, neg_mask, weights, labels, working_dir):
        self.filename = filename
        self.img = img
        self.points = points
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        self.weights = weights  # Sample Weights
        self.labels = labels
        self.working_dir = working_dir


class Keras:
    def __init__(self):
        self.classifier = None
        self.full_model = None

    def train(self, dataset, epochs=60):
        imgs = []
        sample_weights = []
        labels = []
        for labeled_data in dataset:
            sample_weights.append(labeled_data.weights)
            labels.append(labeled_data.labels)
            h, w = labeled_data.img.shape[:2]
            imgs.append(labeled_data.img)

        imgs = np.stack(imgs)
        sample_weights = np.stack(sample_weights)
        labels = np.stack(labels)

        self.classifier, self.full_model = self.create_models(h, w)
        self.full_model.fit(x=(imgs, sample_weights, labels),
                            epochs=epochs)

    def create_models(self, h, w):
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

    def load(self, modelname):
        self.full_model = keras.models.load_model(modelname + ' Model')
        self.classifier = keras.models.load_model(modelname + ' Classifier')
    def load_given(self,dataset):
        model = keras.models.load_model("Given_Model/savedmodel")
        imgs = []
        sample_weights = []
        labels = []
        for labeled_data in dataset:
            sample_weights.append(labeled_data.weights)
            labels.append(labeled_data.labels)
            h, w = labeled_data.img.shape[:2]
            imgs.append(labeled_data.img)

        imgs = np.stack(imgs)
        sample_weights = np.stack(sample_weights)
        labels = np.stack(labels)

        model.fit(x=(imgs),
                            epochs=60)

    def save(self, Prefix="New"):
        self.full_model.save(Prefix + ' Model')
        self.classifier.save(Prefix + ' Classifier')

    def Get_Model(self):
        weights, bias = self.full_model.get_layer('my_dense').get_weights()
        print("Weights", weights, "| Bias", bias)
        self.full_model.summary()

    def predict(self, labeled_data):
        prediction = self.classifier.predict(x=(labeled_data.img[None, :]),verbose = 0)
        pred_probs = prediction[0]
        pred_labels = np.round(pred_probs)  # 1's and 0's
        img_thresh = pred_labels
        return img_thresh

    def validate(self, dataset):
        loss = []
        pix_accuracy = []
        asoc_accuracy = []
        for labeled_data in dataset:
            results = self.full_model.evaluate(
                x=(labeled_data.img[None, :], labeled_data.weights[None, :], labeled_data.labels[None, :]),
                batch_size=1,verbose = 0)
            img_thresh = self.predict(labeled_data)


            # print("For File:", labeled_data.filename)
            # print("Loss:", results[0], ",Accuracy:", results[1])
            # print("-" * 40)

            loss.append(results[0])
            pix_accuracy.append(results[1])

            temporaryLSQ = LSQ()

            img_thresh = np.where(img_thresh > 0, np.uint8(255), np.uint8(0))
            contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            for (i, contour) in enumerate(contours):

                m = cv2.moments(contour)

                area = m['m00']
                if (area == 0): area = 1
                cx = m['m10'] / area
                cy = m['m01'] / area
                centers.append((cx, cy))
            centers = np.array(centers, dtype=np.float32)
            Association_Accuracy = optimze_point_dist(centers, labeled_data, "KERAS")
            asoc_accuracy.append(Association_Accuracy)
            if(Association_Accuracy<0.50):
                print(f"Accuracy Bad {labeled_data.filename}, with Assoc Acc {Association_Accuracy}")
        print(f"Averages:")
        print(f"Average: Loss {self.Average(loss)} | Pix Accuracy {self.Average(pix_accuracy)} | Association Accuracy "
              f"{self.Average(asoc_accuracy)} ")
        print(f"Mins, {min(loss)} | {min(pix_accuracy)} | {min(asoc_accuracy)}")
        print(f"Maxs, {max(loss)} | {max(pix_accuracy)} | {max(asoc_accuracy)}")
    def Average(self,list):
        return (sum(list)/len(list))


class LSQ:
    def __init__(self):
        self.weights = None

    def load(self, filename):
        self.weights = np.loadtxt("LSQ" + filename)

    def loadfromtxt(self, name):
        print(name)
        self.weights = np.loadtxt(name)

    def save(self, filename):
        np.savetxt("LSQ" + filename, (self.weights))

    def train(self, dataset):

        ABig = []
        YBig = []
        all_weights = []

        print("Beginning Training TQDM:")
        for labeled_data in tqdm(dataset):
            megaimg = self.GetMegaImg(labeled_data.img)
            pos_pixels = megaimg[labeled_data.pos_mask]
            neg_pixels = megaimg[labeled_data.neg_mask]
            A = np.vstack((pos_pixels, neg_pixels))
            A_Len = len(A)
            A = np.append(A, np.ones([A_Len, 1]), 1)
            Y = np.ones([A_Len])
            Y[len(pos_pixels):] = -1

            pos_weights = labeled_data.weights[labeled_data.pos_mask]
            neg_weights = labeled_data.weights[labeled_data.neg_mask]
            weights = np.hstack((pos_weights, neg_weights))

            A *= np.sqrt(weights[:, None])  # Generate Augmented Matrix's
            Y *= np.sqrt(weights)

            ABig.append(A)
            YBig.append(Y)
            all_weights.append(weights)

        A = np.vstack(ABig)
        Y = np.hstack(YBig)
        all_weights = np.hstack(all_weights)
        all_weights /= np.sum(all_weights)

        lamda = 1e-5  # Regularization constant
        regularization = lamda * np.identity((A.T @ A).shape[0])
        # print((A.T @ A).shape)
        inverse = np.linalg.inv(A.T @ A + regularization)
        self.weights = inverse @ (A.T @ Y)

        pred = (A @ self.weights) > 0.0
        labels = Y > 0.0
        is_correct = (pred == labels)

        # print('in training accuracy is', is_correct.mean(), 'and weighted accuracy is', (is_correct * all_weights).sum())

    def predict(self, img):  # Create helper for megaimg
        megaimg = self.GetMegaImg(img)
        img_gray = (megaimg * self.weights[:-1]).sum(axis=-1) + self.weights[-1]
        img_thresh = np.where(img_gray > 0, np.uint8(255), np.uint8(0))

        centers = []
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for (i, contour) in enumerate(contours):

            m = cv2.moments(contour)

            area = m['m00']
            if (area == 0): area = 1
            cx = m['m10'] / area
            cy = m['m01'] / area
            centers.append((cx, cy))
        centers = np.array(centers, dtype=np.float32)

        return img_gray, img_thresh, centers  # Syn probability array

    def GetMegaImg(self, img):
        blursizes = [5, 11, 21]
        my_list = [img]
        for size in blursizes:
            current_blur = cv2.blur(img, (size, size))
            my_list.append(current_blur)

        megaimg = np.dstack(my_list)
        return megaimg

    def validate(self, dataset, Print_Mode = 0):  # get Compare center to truth

        # Get img, predict. threshold then use optim
        tally = []
        association = []
        print("Beginning Validation:")
        for labeled_data in tqdm(dataset):
            currimg = labeled_data.img
            img_gray, img_thresh, centers = self.predict(currimg)
            img_thresh = img_thresh.view(bool)

            any_mask = labeled_data.pos_mask | labeled_data.neg_mask
            is_correct = (img_thresh[any_mask] == labeled_data.labels[any_mask])
            accuracy = is_correct.mean()

            weighted_accuracy = (is_correct * labeled_data.weights[any_mask]).sum()
            # print(labeled_data.weights[any_mask].sum())

            if(Print_Mode == 1):
                print("-" * 40)
                print("For File", labeled_data.filename, "results:")
                print("Pixel Based Accuracy is", accuracy, ". Weighted Accuracy is", weighted_accuracy)

            tally.append(accuracy)
            accuracy = optimze_point_dist(centers, labeled_data, type="LSQ",Print_Mode=0)
            association.append(accuracy)
        print("-" * 40)
        tally = sum(tally) / len(tally)
        association = sum(association) / len(association)
        print("Average Pixel Based Accuracy", tally)
        print("Average Association Accuracy", association)

    def Draw(self, labeled_data):
        _, _, centers = self.predict(labeled_data.img)
        img = Draw_Circles(centers, labeled_data)
        cv2.imshow(labeled_data.filename, img)
        cv2.waitKey(0)
        cv2.imwrite("TestFile.jpg", img)
        cv2.destroyAllWindows()


def Draw_Circles(points, labeled_data, existing_img=None):
    if (existing_img == None):
        img = cv2.imread(labeled_data.working_dir + labeled_data.filename)
    else:
        img = existing_img
    for point in points:
        pointx = int(point[0])
        pointy = int(point[1])
        point = (pointx, pointy)
        cv2.circle(img, point, 10, color=(255, 0, 0))
    return img


"""
Optimize Point Distance Given two point sets
input: tuple - points1
       tuple - points2
       string - filename
       string - type - Either "LSQ" or "KERAS"
return: decimal - accuracy
Also saves association diagrams.
"""


def optimze_point_dist(points2, labeled_data, type= 'LSQ', Print_Mode = 0, Maximum_Seperation = 10):
    points1 = labeled_data.points
    filename = labeled_data.filename
    working_dir = labeled_data.working_dir

    img = cv2.imread(working_dir + filename)

    C = cdist(points1, points2)

    assignment1, assigment2 = linear_sum_assignment(C)

    fig, ax = plt.subplots(1, 1)
    ax.plot(points1[:, 0], points1[:, 1], 'bo', markersize=2) # Blue Truth
    ax.plot(points2[:, 0], points2[:, 1], 'rs', markersize=2) # Red Detected
    count = 0

    for (idx1, idx2) in zip(assignment1, assigment2):
        dist = C[idx1, idx2]
        if dist < Maximum_Seperation:
            p1 = points1[idx1]
            p2 = points2[idx2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')
            count += 1

    plt.imshow(img[:, :, ::-1], interpolation='none', filternorm=False)
    (base, _) = os.path.splitext(filename)

    accuracy = count / (len(points1) + len(points2) - count)
    if(Print_Mode == 1):
        print("For", filename, "", type, "Has Associated ", count, "Points out of a total", len(points2),
              "detected points and", len(points1), "truth points with accuracy", accuracy)

    plt.title(("Accuracy", accuracy))
    base, _ = os.path.splitext(labeled_data.filename)
    base.replace('_', ' ')
    if type == "LSQ":
        plt.savefig("AssociationDiagrams/" + base + " LSQ.pdf", format="pdf")
    if type == "KERAS":
        plt.savefig("AssociationDiagrams/" + base + " Keras.pdf", format="pdf")

    plt.clf()
    plt.close(fig)
    return accuracy


"""
Function Load Data
input: string - filename 
return: dataset object
"""


def load_data(filenames, working_dir=''):
    dataset = []
    print("Loading Files TQDM:")
    for imgfile in tqdm(filenames):
        # Load All nessesary components for class
        img, points = Read_Images(imgfile, working_dir)
        pos_mask, neg_mask = Generate_Masks(img, points)
        pos, neg, weights, labels = Generate_Weights(img, pos_mask, neg_mask)
        # pos,neg,sample.label
        obj = labeled_data(imgfile, img, points, pos_mask, neg_mask, weights, labels, working_dir)
        dataset.append(obj)

    return dataset


"""
Function Read_Image
input: tuple - imgfiles - containing imgfiles
return: np array - img - normalized
        np array - x - list of x truth
        np array - y - list of y truth
"""


def Read_Images(imgfile=None, workingdir=''):
    (base, ext) = os.path.splitext(imgfile)
    # print('base:', base, 'ext:', ext)
    txtfile = workingdir + base + ".txt"
    imgfile = workingdir + imgfile

    read = cv2.imread(imgfile)
    assert read is not None
    read = read.astype(np.float32)
    read -= read.mean(axis=(0, 1))
    read /= read.std(axis=(0, 1))

    points = np.genfromtxt(txtfile, delimiter=' ', dtype=float)

    return read, points


"""
Function Generate_Masks
input: object - labeled_data - 
output: np array - pos_mask
        np array - neg_mask
return: 
"""


def Generate_Masks(img, points):
    h, w = img.shape[:2]
    pos_mask = np.zeros((h, w), np.uint8)  # 1000x1000 black / 0
    neg_mask = np.full((h, w), 255, np.uint8)  # 1000x1000 white / 255

    for x, y in points.astype(int):  # Drawing Masks
        pos_mask = cv2.circle(img=pos_mask, center=(x, y)
                              , radius=3, color=(255, 255, 255), thickness=-1)
        neg_mask = cv2.circle(img=neg_mask, center=(x, y)
                              , radius=7, color=(0, 0, 0), thickness=-1)
    pos_mask = pos_mask.astype(bool)  # Making Masks
    neg_mask = neg_mask.astype(bool)
    return pos_mask, neg_mask


"""
Function Generate_Weights
input: np array - img 
       np array - pos_mask bool
       np array - neg_mask - bool 
output: constants:
        pos_weight
        neg_weight
        sample_weight
        labels 
"""


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

    return pos_weight, neg_weight, sample_weight, labels


"""
Function Print Filenames
input: object list - dataset - 
return: None
prints out filenames 
"""


def print_filenames(dataset):
    print("Dataset Length", len(dataset))
    for i, labeled_data in enumerate(dataset):
        print("File", i, ":", labeled_data.filename)
    print("-" * 40)


"""
Driver reset. Ill keep this just for reference. Shoulnt be used
"""
# def main():
#     imgfiles = []  # Load Set Of All Tif Files
#     os.chdir(r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\Classes")
#     for file in glob.glob("*.tif"):
#         imgfiles.append(file)
#     imgfiles = ["test_1000_6000.tif", "test_11000_10000.tif", "test_2000_0.tif"]
#     # imgfiles = ["2.jpg"]
#
#     testing_set = load_data(imgfiles)
#     training_set = testing_set[:2]
#
#     print("Training Set:")
#     print_filenames(training_set)
#     print("Testing Set:")
#     print_filenames(testing_set)
#
#     my_obj = LSQ()
#     my_obj.train(training_set)
#     my_obj.save("Weights.txt")
#     my_obj.validate(testing_set)
#
#     my_keras = Keras()
#     my_keras.load("WholeSet")
#     # my_keras.Get_Model()
#     my_keras.validate(testing_set)


if __name__ == "__main__":
    main()
