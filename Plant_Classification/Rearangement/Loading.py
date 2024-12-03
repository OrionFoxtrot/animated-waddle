from tensorflow import keras
import My_Package as mp
import cv2
import numpy as np
import os
import tensorflow as tf



def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    full_model = keras.models.load_model('MyModel')
    classifier_only = keras.models.load_model('MyClassifier')
    print("Loaded Model")

    #imgfile = "test_11000_10000.tif"
    #imgfile="test_1000_6000.tif"
    imgfile = "test_1000_6000.tif"
    Eval_img, Eval_x, Eval_y = mp.Read_Image(imgfile)
    Eval_pos_mask, Eval_neg_mask = mp.Generate_Masks(Eval_x, Eval_y)
    Eval_sample_weight, Eval_any_mask, Eval_labels,_,_\
        = mp.Generate_Weights(Eval_img, Eval_pos_mask, Eval_neg_mask)

    results = full_model.evaluate(x=(Eval_img[None, :], Eval_sample_weight[None, :], Eval_labels[None, :]),
                                  batch_size=1)
    print("test loss, test accuracy:", results)
    weights, bias = full_model.get_layer('my_dense').get_weights()
    print("W",weights,"B",bias)

    Eval_img, Eval_x, Eval_y = mp.Read_Image(imgfile)
    prediction = classifier_only.predict(x=(Eval_img[None, :]))

    #type 1 = minmax, type 0 = drawcontour, lsq = 0 no lsq, lsq = 1 yes lsq
    img_gray,img_thresh,centers = mp.convert(prediction,imgfile="test_11000_10000.tif",imgfile1 ="test_1000_6000.tif", type = 0,lsq=1)



    Eval_img = cv2.imread(imgfile)

    for cx,cy in centers:
        cv2.circle(Eval_img, (int(cx * 8), int(cy * 8)), 25, (0, 0, 255),
                   thickness=1, lineType=cv2.LINE_AA, shift=3)
    for x,y in zip(Eval_x,Eval_y):
        cv2.circle(Eval_img, (int(x * 8), int(y * 8)), 25, (0, 255, 255),
                   thickness=1, lineType=cv2.LINE_AA, shift=3)

    temp = np.dstack((Eval_x,Eval_y))[0,:,:]
    mp.optimze_point_dist(temp,centers,Eval_img)

    mp.display_results([Eval_img,img_gray])




if __name__ == "__main__":
    main()