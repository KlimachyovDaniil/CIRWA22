# card_processor.py
import numpy as np
import math
import cv2
import cv2 as cv
from operator import mul
from functools import reduce

def compute_square_quad(box):
    def compute_dist(p1, p2):
        return sum([(d1 - d2) ** 2 for d1,d2 in zip(p1, p2)]) ** 0.5

    def compte_square_tri(p1, p2, p3):
        a = compute_dist(p1, p2)
        b = compute_dist(p2, p3)
        c = compute_dist(p1, p3)
        p = (a + b + c) / 2
        return math.sqrt(p * (p - a) * (p - b) * (p - c))

    p1, p2, p3, p4 = box
    return compte_square_tri(p1, p2, p3) + \
          compte_square_tri(p1, p4, p3)

def choose_card_box(img, boxes, eps = 0.01):
    img_square = reduce(mul, img.shape[:2], 1)

    square_box_list = []

    for box in boxes:
        square_box_list.append([compute_square_quad(box), box])

    square_box_list.sort(key=lambda x: x[0], reverse=True)
    square_box_list[0]

    for square, box in square_box_list:
        if 1 - square / img_square > eps:  # exclude picture border
            # cv.drawContours(img,[box],0,(255,0,0),2) # рисуем прямоугольник
            # cv2_imshow(img)   
            return box

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def get_card_img(img):
    hsv_min = np.array((0, 54, 5), np.uint8)
    hsv_max = np.array((187, 255, 253), np.uint8)

    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    boxes = []
    # img2show = img.copy()
    for i, cnt in enumerate(contours0):
        rect = cv.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        boxes.append(box)
        # cv.drawContours(img2show,[box],0,(255,0,0),2) # рисуем прямоугольник
        
    # cv2_imshow(img2show) # вывод обработанного кадра в окно

    card_box = choose_card_box(img, boxes, eps = 0.01)
    return four_point_transform(img, card_box), card_box
