from attr import NOTHING
import cv2
import numpy as np

def nothing(x):
     pass

cv2.namedWindow("Canny_Trackbar") 
cv2.createTrackbar("Threshold 1", "Canny_Trackbar", 0, 255, nothing)
cv2.createTrackbar("Threshold 2", "Canny_Trackbar", 0, 255, nothing)

while 1:

 img = cv2.imread('gray_image_sim_new.png')
 original_img = img.copy()

 thresh1 = cv2.getTrackbarPos("Threshold 1", "Canny_Trackbar")
 thresh2 = cv2.getTrackbarPos("Threshold 2", "Canny_Trackbar")
 gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
 blur_image = cv2.GaussianBlur(gray_image, (5,5), 0) 
 canny_image = cv2.Canny(blur_image, thresh1, thresh2)
 cv2.imshow("Canny image", canny_image)
 cv2.waitKey(300)

