
import rospy
import cv2
import matplotlib.pylab as plt
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


bridge = CvBridge()
command_controller =rospy.Publisher("motor_commands", String, 100)

def nothing(x):
     pass

cv2.namedWindow("Canny_Trackbar") 
cv2.createTrackbar("Threshold 1", "Canny_Trackbar", 20, 255, nothing)
cv2.createTrackbar("Threshold 2", "Canny_Trackbar", 42, 255, nothing)


def draw_avgline(image, line_parameters):   #making co ordinates for avg line
      slope, intercept = line_parameters
      y1 = image.shape[0]
      y2= int(y1//2.5)
      x1 = int((y1 - intercept)/slope)
      x2 = int((y2 - intercept)/slope)
      return np.array([x1, y1, x2, y2])


def average_line(image,lines):         #averaging Hough lines
      left_fit= []
      right_fit= []
      for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters= np.polyfit((x1,x2),(y1,y2),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope>0:
                  left_fit.append((slope,intercept))
            else:
                  right_fit.append((slope,intercept))
      left_fit_avg= np.average(left_fit, axis=0)
      right_fit_avg=np.average(right_fit, axis=0)
      left_line = draw_avgline(image, left_fit_avg)
      right_line = draw_avgline(image, right_fit_avg)
      #print(right_fit_avg)
      #print(left_fit_avg)

      return np.array([left_line,right_line])



def motor_plan(left,right):                     #Motor planning method
   command = "Stop"
   black_value = 25                             #setting a black color value in gray scale.
   if left < black_value and right < black_value:
        print("straight")
        command = "S"
   if left > black_value and right <black_value:
        print("Turn Right")
        command = "R"
   if left <black_value and right >black_value:
        print("Turn Left")
        command = "L"
  
   command_controller.publish(command)   

def canny(image): 
  blur_image = cv2.GaussianBlur(image, (5,5), 0)    # Converting gray image to blur image for noise reduction
  thresh1 = cv2.getTrackbarPos("Threshold 1", "Canny_Trackbar")
  thresh2 = cv2.getTrackbarPos("Threshold 2", "Canny_Trackbar")
  canny_image = cv2.Canny(image, thresh1, thresh2)
  return canny_image




def region_of_interest(image, vertices):          #Creating region of interest 
  mask = np.zeros_like(image)
  match_mask_color = 255               #set mask color
  cv2.fillPoly(mask, vertices, match_mask_color)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image





def draw_lines(image,lines):                      #Drwaing Hough lines in the image
      
  img= np.copy(image)
  line_image= np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)
  if lines is not None:
   for line in lines:
        x1, y1, x2, y2  = line.reshape(4)
        cv2.line(line_image,(x1,y1),(x2,y2),(100,255.100), 5)

  img = cv2.addWeighted(img, 0.8, line_image, 1, 0.0)
  return img




def laneDetector(data):                                 #ROS callback function to the /camera/image_raw topic
  cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
  gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY) 
  motor_plan(gray_image[770][280], gray_image[770][520])
  #print(gray_image[775][275], gray_image[775][530])
  #gray_image =cv2.line(gray_image, (280,775),(560,775) ,80 ,5)
  #print(cv_image.shape)
  height = cv_image.shape[0]
  canny_image = canny(gray_image)
  region_of_interest_points = [ (100,height) ,(400,420),(450,420),(650,800)]  #finding the region of interest parameter value
  cropped_image = region_of_interest(canny_image, np.array([region_of_interest_points], np.int32)) #cropping the image according to our region of interest
  lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/180, threshold=40, lines= np.array([]), minLineLength=10, maxLineGap=100) #Hough lines parameters
  avg_lines= average_line(cropped_image,lines)
  hough_image= draw_lines(cv_image,avg_lines)
  newline_image =cv2.line(hough_image, (280,770),(520,770) ,255 ,5)
  #motor_plan(gray_image[775][275], gray_image[775][530])
  #print(gray_image[775][275], gray_image[775][530])



  #for i in range(0,800):
       #print(i,canny_image[790][i])
  #for i in range(550,630):
       #print(i,gray_image[790][i])

  #cv2.imshow("Simulated Image",cv_image)
  cv2.imshow("Gray image",gray_image)
  cv2.imshow("houghline_image", hough_image)
  cv2.imshow("Canny Image", canny_image)
  cv2.imshow("cropped Image", cropped_image)
  cv2.waitKey(3)



  #plt.imshow(gray_image)
  #plt.imshow(cv_image)
  #plt.show()
  #cv2.line(cv_image, (100,50))
 






def main():                                           #Main method
  print("Helloooo World")
  rospy.init_node('my_planner')
  topic_name = "/camera/image_raw"
  img_sub = rospy.Subscriber(topic_name, Image, laneDetector)
  rospy.spin()

if __name__ == "__main__":
  main()

