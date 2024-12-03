
import numpy as np
import cv2

if __name__ == '__main__':
    print("Main First Pass")

    #image = cv2.imread(r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\EvenSmallerSlices\test_0_0_200_400.tif") #Open Img
    image = cv2.imread(r"C:\Users\lohat_jay97s3\OneDrive\Desktop\School\Projects\SB3 PyCharm\SR\Plants!\SlicedImage\test_1000_6000.tif")

    cv2.imwrite("SaveOut\Original.png", image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert grey
    cv2.imwrite("SaveOut\Grey.png", grey)

    medianA = cv2.medianBlur(grey, 1) # Median Filt from grey
    cv2.imwrite("SaveOut\MedianA.png", medianA)

    img_binary = cv2.threshold(medianA, 150, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite("SaveOut\Img_Bin.png", img_binary)

    medianB = cv2.medianBlur(img_binary, 5)  # Median Filt from grey
    cv2.imwrite("SaveOut\MedianB.png", medianB)

    cv2.imshow("Original", medianB)
    # cv2.imshow("Keypoints", circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    medianBFlip = cv2.bitwise_not(medianB)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closing = cv2.morphologyEx(medianBFlip, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("SaveOut\closing.png", closing)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))

    for (i, contour) in enumerate(contours):

        m = cv2.moments(contour)

        area = m['m00']
        if(area==0): area=1
        cx = m['m10'] / area
        cy = m['m01'] / area

        print(f'  - contour {i} has {len(contour)} points, area={area}, center at ({cx}, {cy})')
        cv2.circle(image, (int(cx * 8), int(cy * 8)), 25, (0, 0, 255),
                   thickness=1, lineType=cv2.LINE_AA, shift=3)

    cv2.imwrite("SaveOut\SO.png",image)
    cv2.imshow("Original",image)
    #cv2.imshow("Keypoints", circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)



