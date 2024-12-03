import cv2
import numpy as np
import os.path
from matplotlib import pyplot as plt


def main():
    print("Hello World")
    imgfile = "test_1000_6000.tif"

    #imgfile = "test_11000_10000.tif"
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

    blursizes = [5, 11, 21]
    #blursizes=[]
    my_list = [img]

    for size in blursizes:
        current_blur = cv2.blur(img, (size, size))
        my_list.append(current_blur)

    megaimg = np.dstack(my_list)
    print("Mega Img Shape", megaimg.shape)
    pos_pixels = megaimg[pos_mask]
    neg_pixels = megaimg[neg_mask]
    A = np.vstack((pos_pixels, neg_pixels))

    A_Len = len(A)

    A = np.append(A, np.ones([A_Len, 1]), 1)
    Y = np.ones([A_Len])
    Y[len(pos_pixels):] = -1

    print("A Shape", A.shape, ",Y Shape", Y.shape)
    print(A)
    # -----E:GENERATE MASKS-----#

    # ------S:GENERATE WEIGHTS-----#

    WiP = 0.5 / len(pos_pixels)
    WiN = 0.5 / len(neg_pixels)
    print("Pos Weights", WiP)
    print("Neg Weights", WiN)

    w = np.zeros(len(A))
    w[0:len(pos_pixels)] = WiP
    w[len(pos_pixels):] = WiN
    print('w total is', w.sum(), "aprox = 1")
    print(Y)

    A *= np.sqrt(w[:, None])  # Generate Augmented Matrix's
    Y *= np.sqrt(w)

    # -----E:GENERATE WEIGHTS-----#

    # -----S:Solver-----#
    lamda = 1e-5  # Regularization constant
    regularization = lamda * np.identity((A.T @ A).shape[0])
    print((A.T@A).shape)
    inverse = np.linalg.inv(A.T @ A + regularization)
    Sol = inverse @ (A.T @ Y)
    np.savetxt("LeastSquaresSol.txt",Sol)


    # Input Manual Sol
    #Sol = np.array([0.16149866, -0.30169163, 0.23941726, 0.56642764, -0.03535925, -1.0117191,
     #       - 1.68478797, 2.75605113, -1.61645888, -0.01114622, -1.01705296, 1.97009945,
      #      -0.59155838])

    print("X constants:", Sol)
    print("Solutions Shape", Sol.shape)
    # -----E:Solver-----#

    # -----S: Evaluation-----#
    Check = np.sign(A @ Sol)
    print("Weighted Accuracy:", (w * (Check == np.sign(Y))).sum())
    print("Unweighted Accuracy:", (Check == np.sign(Y)).mean())

    constants = (len(blursizes) + 1) * 3
    img_gray = (megaimg * Sol[:constants]).sum(axis=-1) + Sol[constants]  # 3 becomes n constants, or chanels.

    img_thresh = np.where(img_gray > 0, np.uint8(255), np.uint8(0))

    imin = img_gray.min()
    imax = img_gray.max()

    img_gray = (img_gray - imin) / (imax - imin)

    print("ImgGray", img_gray.shape, img_gray.dtype)
    print("ImgGrayMinMax", img_gray.min(), img_gray.max())


    img = cv2.imread(imgfile)
    mask = img_thresh.astype(bool)
    img[~mask] = 0
    cv2.imwrite("OVerlay.png",img)
    #overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("img_thresh.png",img_thresh)
    img_gray = (img_gray * 255).astype(np.uint8)
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB)
    for index in range(len(x)):

        img_thresh = cv2.circle(img=img_thresh, center=(x[index], y[index])
                                , radius=3, color=(255, 0, 255), thickness=-1)

    cv2.imshow("Threshold", img_thresh)
    cv2.imshow("Original Blur", img_gray)

    cv2.imwrite("img_gray.png",img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    fig = plt.figure(figsize=(10, 5))
    #plt.bar(np.arange(len(Sol) - 1), Sol[:-1], color='blue',
            #width=0.4)
    plt.xticks(np.arange(len(Sol)))
    plt.bar(np.arange(len(Sol)),Sol)
    plt.ylabel("Weighting (n)")
    #plt.show()
    plt.grid()


    plt.savefig("myplotsave.png")

    # -----E:Evaluation-----#

    #Hellowwordlkeras

if __name__ == '__main__':
    main()
