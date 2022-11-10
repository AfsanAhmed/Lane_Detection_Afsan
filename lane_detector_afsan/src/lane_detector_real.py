import cv2
import matplotlib.pylab as plt
import numpy as np



mask_color =255
def nothing(x):
     pass

cv2.namedWindow("Canny_Trackbar") 
cv2.createTrackbar("Threshold 1", "Canny_Trackbar", 50, 255, nothing)
cv2.createTrackbar("Threshold 2", "Canny_Trackbar", 150, 255, nothing)


def edgeDetector(image):
  blur_image = cv2.GaussianBlur(image, (5,5), 0)    # Converting image to blur image
  thresh1 = cv2.getTrackbarPos("Threshold 1", "Canny_Trackbar")
  thresh2 = cv2.getTrackbarPos("Threshold 2", "Canny_Trackbar")
  canny_image = cv2.Canny(blur_image, thresh1,thresh2)  
  #canny_image = cv2.Canny(blur_image, 130, 200)              #Using canny to detect edges   
  return canny_image

def region_of_interest(image, vertices):          #Creating region of interest (which area we are taking in consideratino to detect lanes)
  mask = np.zeros_like(image)
  #channel_count = image.shape[2]
  match_mask_color = mask_color              #* channel_count
  cv2.fillPoly(mask, vertices, match_mask_color)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

def draw_lines(image,lines):                      #Drwaing Hough lines in the image
      
  img= np.copy(image)
  line_image= np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)
  if lines is not None:
   for line in lines:
        x1, y1, x2, y2  = line.reshape(4)
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0), 12)

  img = cv2.addWeighted(img, 0.8, line_image, 1, 0.0)
  return img


def make_coordinates(image, line_parameters):
      slope, intercept = line_parameters
      y1 = image.shape[0]
      y2=  y1//3
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
      return np.array([x1, y1, x2, y2])


def average_slope_intercept(image,lines):
     left_fit= []
     right_fit= []
     for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope < 0:
                  left_fit.append((slope, intercept))
            else:
                  right_fit.append((slope, intercept))
     left_fit_avg = np.average(left_fit, axis=0)
     right_fit_avg=np.average(right_fit, axis=0)
     left_line = make_coordinates(image, left_fit_avg)
     right_line = make_coordinates(image, right_fit_avg)
     #print(right_fit_avg, "right")
     #print(left_fit_avg, "left")

     return np.array([left_line,right_line])
#"""

while 1 :
  image=cv2.imread('test000.jpg')
  height = image.shape[0]
  width = image.shape[1]
  gray_image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
  canny_image = edgeDetector(gray_image)
  region_of_interest_points = [ (0,height) , ((width//2),(height//5)) ,( width ,height)]  #finding the region of interest parameter value
  cropped_image = region_of_interest(canny_image, np.array([region_of_interest_points], np.int32)) #cropping the image according to our region of interest
  lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=60, lines= np.array([]), minLineLength=60, maxLineGap=150) #Hough lines parameters
  avg_lines = average_slope_intercept(image,lines)
  image_with_lines= draw_lines(image , avg_lines)

  #print(image.shape)
  #for i in range(0,1280):
   # print(i,canny_image[719][i])
    
      
      
  #print(gray_image[200][700])
  #print(canny_image.shape)



  #plt.imshow(gray_image)
  #plt.imshow(cv_image)
  #plt.show()
  #print(image.shape)
  cv2.imshow("Gray Image",gray_image)
  cv2.imshow("houghlines Image",image_with_lines)
  cv2.imshow("Cropped Image",cropped_image)
  cv2.imshow("Canny Image",canny_image)
  #cv2.imshow("Raw Image",image)
  cv2.waitKey(300)

#"""





#Lane detection from Vedio
"""
video = cv2.VideoCapture("newvideo2.mp4")   # capturing video 

while(video.isOpened()):
     _, frame =  video.read()               #Taking each frame from the video as an image
     canny_image=canny(frame)               #Calling Canny method 
     region_of_interest_points = [ (0,720) , (0,400), (950,170) ,(1280,720)]  #Setting region of interest parameter.

     cropped_image=region_of_interest(canny_image,np.array([region_of_interest_points], np.int32))  #Cropping the image 
     lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=60, lines= np.array([]), minLineLength=40, maxLineGap=50) #Hough lines parameters
     #avg_lines= average_slope_intercept(frame,lines)
     image_with_lines= draw_lines(frame,lines)
     cv2.imshow("frame",image_with_lines)
     cv2.waitKey(3)
     #if cv2.waitKey(1) & 0XFF == ord('q'):                         #loop breaks when video is finished
      #  break
video.release() 
cv2.destroyAllWindows()                                              #destoying all windows 
"""