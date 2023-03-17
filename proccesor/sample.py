import cv2
from card_processor import get_card_img

img = cv2.imread('sample/im1.jpg')
cv2.imshow('1', get_card_img(img))
cv2.waitKey(0)

img = cv2.imread('sample/im2.jpg')
cv2.imshow('2', get_card_img(img))
cv2.waitKey(0)

img = cv2.imread('sample/im3.jpg')
cv2.imshow('3', get_card_img(img))
cv2.waitKey(0)
