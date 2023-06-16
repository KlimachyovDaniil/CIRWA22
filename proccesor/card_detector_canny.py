import cv2
import numpy as np
from number_recognizer import recognize
from card_detection import get_card_img


def find_contours_of_card_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    cntrs, _ = cv2.findContours(image=dilate, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    return cntrs


def find_coordinates(cntrs, img):
    img_crop = []
    for i in range(len(cntrs)):
        x, y, w, h = cv2.boundingRect(cntrs[i])
        if img.shape[0] * img.shape[1]/5 < w * h < img.shape[0] * img.shape[1]:
            img_crop = img[y :y + h, x :x + w]
    return img_crop


def get_card_img_canny(img):
    cnts = find_contours_of_card_canny(img)
    return find_coordinates(cnts, img)


if __name__ == "__main__":
    img = cv2.imread("../input/img11.jpeg")
    res = get_card_img(img)[0]
    rec, card_num = recognize(res)
    if len(card_num) <= 8:
        res = get_card_img_canny(img)
        rec, card_num = recognize(res)
    cv2.imshow('fff', rec)
    cv2.waitKey(0)
