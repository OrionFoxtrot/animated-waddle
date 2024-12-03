# have pix , use for negatives
# idx = np.arrange(len(pix))
# np.random.shuffle(idx)
# subset=pixels[idx[:5000]]
# OR just do np.random.choice(Args)

# positives white on black back
# negative is black on white back
import numpy as np
import csv
import cv2
import matplotlib
from matplotlib import pyplot as plt


def main():
    # open image
    img = cv2.imread("test_1000_6000.tif")
    #normalize img on read. subtract per chanel mean and div per chanel stdv
    img = img.astype(np.float32)
    img -= img.mean(axis=(0,1))
    img /= img.std(axis=(0,1))

    x, y = [], []
    with open('test_1000_6000.txt', 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            x.append(int(float(row[0])))
            y.append(int(float(row[1])))
    #np.genfromtxt

    pos_mask = np.zeros((1000, 1000), np.uint8)  # 1000x1000 black / 0
    neg_mask = np.full((1000, 1000), 255, np.uint8)  # 1000x1000 white / 255

    x, y = np.array(x), np.array(y)
    print(len(x), "=", len(y))
    for index in range(len(x)):
        # print((x[index],y[index]))
        pos_mask = cv2.circle(img=pos_mask, center=(x[index], y[index])
                              , radius=3, color=(255, 255, 255), thickness=-1)
        neg_mask = cv2.circle(img=neg_mask, center=(x[index], y[index])
                              , radius=7, color=(0, 0, 0), thickness=-1)

    cv2.imwrite("Pos_Mask.png", pos_mask)
    cv2.imwrite("Neg_Mask.png", neg_mask)

    pos_mask = pos_mask.astype(bool)
    neg_mask = neg_mask.astype(bool)
    print(pos_mask.sum())
    print(neg_mask.sum())
    # print(pos_mask.shape,neg_mask.shape)
    # ------------MASKS MADE----------------#

    #-------------POS/NEG------------------#
    #11,22,33 - 92,95
    #5,11,16 - 92,95
    #22,44,66 - 89,95
    #5,11,21 - 93,95

    # TODO: blur_sizes = [5, 11, 21]
    # TODO: mega_img first, then mask so only need to apply mask once

    imgblurS = cv2.blur(img, (5, 5))
    imgblurM = cv2.blur(img, (11, 11))
    imgblurL = cv2.blur(img, (21, 21))

    print("SHAPE",imgblurS.shape)

    pos_pixelsOrg = (img[pos_mask].astype(float))
    neg_pixelsOrg = (img[neg_mask].astype(float))

    pos_pixelsS = (imgblurS[pos_mask].astype(float))
    pos_pixelsM = (imgblurM[pos_mask].astype(float))
    pos_pixelsL = (imgblurL[pos_mask].astype(float))

    neg_pixelsS = (imgblurS[neg_mask].astype(float))
    neg_pixelsM = (imgblurM[neg_mask].astype(float))
    neg_pixelsL = (imgblurL[neg_mask].astype(float))
    # ---------------POS/NEG------------------#



    # # select neg pixels NO MORE SELECTION
    # idx = np.arange(len(neg_pixelsOrg))
    # np.random.shuffle(idx)
    # # neg_pixels=neg_pixels[idx[0:1*len(pos_pixels)]] # num of negatives seleted
    # # neg_pixels = neg_pixels[:len(pos_pixels)]

    #print("Final Dim, Pos:", pos_pixelsOrg.shape, "Neg:", neg_pixelsOrg.shape)

    A = np.vstack((pos_pixelsOrg, neg_pixelsOrg))
    As = np.vstack((pos_pixelsS, neg_pixelsS))
    Am = np.vstack((pos_pixelsM, neg_pixelsM))
    Al = np.vstack((pos_pixelsL, neg_pixelsL))

    print("Old Dim A:", A.shape)
    A = np.hstack((A,As,Am,Al))
    #A = np.dstack((A,As,Am,Al))
    print("New Dim A:", A.shape)
    print(A)

    # A.shape[0] -> len(A)
    A = np.append(A, np.ones([A.shape[0], 1]), 1)
    print("Added Drift A:", A.shape)

    Y = np.ones([A.shape[0]])

    # num_pos = len(pos_pixel...)
    Y[len(pos_pixelsOrg):] = -1  # created Y with half pos, half neg.
    print("YS",Y.shape)

    #-------------WEIGHTS MAKE--------------------#
    # Make Weights
    WiP = 0.5 / len(pos_pixelsOrg)
    WiN = 0.5 / len(neg_pixelsOrg)
    print("Pos Weights", WiP)
    print("Neg Weights", WiN)

    w = np.zeros(len(A))
    w[0:len(pos_pixelsOrg)] = WiP
    w[len(pos_pixelsOrg):] = WiN

    # idx_shifted = idx + len(pos_pixels)
    # w[len(pos_pixels)*2:] = 0.0

    print('2 * pos len = ', 2 * len(pos_pixelsOrg))
    print('w total is', w.sum())

    # A[:len(pos_pixels)] *= np.sqrt(WiP)
    # Y[:len(pos_pixels)] *= np.sqrt(WiP)
    # A[len(pos_pixels):] *= np.sqrt(WiN)
    # Y[len(pos_pixels):] *= np.sqrt(WiN)

    A *= np.sqrt(w[:, None])
    Y *= np.sqrt(w)

    #--------------END WEIGHTS MAKE--------------#

    # foo, = np.nonzero(w)
    # A = A[foo]
    # Y = Y[foo]
    # w = w[foo]

    # A = np.vstack((A, np.zeros((40000, 4))))
    # Y = np.hstack((Y, np.zeros(40000)))
    # w = np.hstack((w, np.zeros(40000)))

    # solve using A,Y
    lam = 1e-5
    addedC = lam * np.identity((A.T @ A).shape[0])
    inverse = np.linalg.inv(A.T @ A + addedC)

    Sol = inverse @ (A.T @ Y)
    print("X:", Sol)
    print("Sol Shape,",Sol.shape)

    # check
    Check = np.sign(A @ Sol)
    print("Weighted Accuracy:", (w * (Check == np.sign(Y))).sum())
    print("Unweighted Accuracy:", (Check == np.sign(Y)).mean())

    megaimg = np.dstack((img,imgblurS,imgblurM,imgblurL))
    print("Mega Img Shape",megaimg.shape)

    # 12 -> num_channels
    img_gray = (megaimg * Sol[:12]).sum(axis=-1) + Sol[12]  # 3 becomes n constants, or chanels.

    img_thresh = np.where(img_gray > 0, np.uint8(255), np.uint8(0))

    imin = img_gray.min()
    imax = img_gray.max()

    img_gray = (img_gray - imin) / (imax - imin)

    print("ImgGray", img_gray.shape, img_gray.dtype)
    print("ImgGrayMinMax", img_gray.min(), img_gray.max())

    img_gray = (img_gray * 255).astype(np.uint8)

    # img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.THRESH_BINARY,
    #                          cv2.THRESH_BINARY, 11, 7)

    # ret, img_thresh = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY)

    #img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
     #                                  cv2.THRESH_BINARY, 21, -12)

    # img_gray=cv2.GaussianBlur(img_gray,(5, 5),0)
    # ret2, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # img_thresh = cv2.bitwise_not(img_thresh)

    #img_thresh = cv2.medianBlur(img_thresh, 5)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Morph. X
    # losing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB)
    for index in range(len(x)):
        # print("Draw",index)
        img_thresh = cv2.circle(img=img_thresh, center=(x[index], y[index])
                                , radius=3, color=(255, 0, 255), thickness=-1)

    cv2.imshow("Thresh", img_thresh)
    cv2.imshow("Gray", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(np.arange(len(Sol)-1),Sol[:-1], color='maroon',
            width=0.4)
    plt.show()

    # cv2.imwrite("Final_Out.png",img_gray)
    # #scikit.learn

    # support vector machine
    #V keras


if __name__ == '__main__':
    main()
