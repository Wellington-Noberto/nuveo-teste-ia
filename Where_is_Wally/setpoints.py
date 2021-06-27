import cv2
import numpy as np
import os


def empty(a):
    pass


def setpoints():

    cv2.namedWindow("Parameters")
    cv2.resizeWindow('Parameters', 640, 240)
    cv2.createTrackbar('Threshold1', "Parameters", 150, 255, empty)
    cv2.createTrackbar('Threshold2', "Parameters", 255, 255, empty)


    while True:
        img = cv2.imread('data/01-WheresWally/TrainingSet/wally_001.jpg')
        img_blur = cv2.GaussianBlur(img, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        threshold1 = cv2.getTrackbarPos('Threshold1', "Parameters")
        threshold2 = cv2.getTrackbarPos('Threshold2', "Parameters")

        img_canny = cv2.Canny(img_gray, threshold1, threshold2)
        img_thresh = cv2.adaptiveThreshold(img_canny, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #ret, img_thresh = cv2.threshold(img_gray, 127, 255, 0)

        img_stack = np.hstack((img_gray, img_thresh, img_canny))

        cv2.imshow('setpoints', img_stack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break


def test_images():
    img_paths = os.listdir('data/01-WheresWally/TrainingSet/')

    for img_path in img_paths:
        if img_path.endswith('.jpg'):
            img = cv2.imread('data/01-WheresWally/TrainingSet/' + img_path)

            img_blur = cv2.GaussianBlur(img, (5, 5), 1)
            img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
            #img_canny = cv2.Canny(img_gray, 150, 200)
            # dilate and erode
            img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


            # Contours
            contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 10000 <= area <= 100000:
                    M = cv2.moments(cnt)

                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(img, (cX, cY), 5, (0, 255, 255), -1)
                    cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow('teste', img_thresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)
            cv2.destroyAllWindows()




if __name__ == '__main__':
    #setpoints()
    test_images()
