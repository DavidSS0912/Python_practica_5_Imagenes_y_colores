import cv2
import numpy as np


def getMask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV (hue,saturation, value)
    lower_red = np.array([130, 100, 100], np.uint8)
    upper_red = np.array([190, 255, 255], np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return mask


def run():
    cap = cv2.VideoCapture(0)

    while(1):
        _, frame = cap.read()
        webCam = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)

        mask = getMask(webCam)
        mask = cv2.medianBlur(mask, 13)
        mask = cv2.bitwise_not(mask)
        res = cv2.bitwise_and(webCam, webCam, mask=mask)

        res[np.where((res == [0, 0, 0]).all(axis=2))] = [255, 50, 0]

        res = cv2.bitwise_and(webCam, res)

        cv2.imshow('Cambio del color rojo', res)
        cv2.imshow("Mask", mask)
        cv2.imshow('WebCam', webCam)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if "__main__" == __name__:
    run()
