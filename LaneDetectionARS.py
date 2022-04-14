import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("Video/input_video.mp4")

width = 350
height = 500

size = (width, height)

result = cv2.VideoWriter('Video/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while True:
    ret, frame = cap.read()
    
    if ret == True: 
        
        # bird eye view transform
        actual_points = np.float32([[0,400],[0,640],[640,400],[640,640]])
        transform_points = np.float32([[width,0],[width,height],[0,0],[0,height]])
        matrix = cv2.getPerspectiveTransform(actual_points,transform_points)
        transformed = cv2.warpPerspective(frame, matrix, (width, height))
        
        # image blurring for noise removal
        gray_img = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
        
        # thresholding image 
        retval, threshold = cv2.threshold(blur_img, 205, 255, cv2.THRESH_BINARY)
        
        # canny edge detection
        canny_edge_image = cv2.Canny(threshold, 280, 300)
        
        # probabilistic hough line transform 
        rho = 2
        theta = np.pi / 180
        threshold = 15
        min_line_length = 20
        max_line_gap = 100
        
        lines = cv2.HoughLinesP(canny_edge_image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        
        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(transformed, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv2.LINE_AA)
        cv2.imshow("Hough Line Transformed", transformed)
  
        # save result video
        result.write(transformed)
  
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
  
    else:
        break
    
cap.release()
result.release()
cv2.destroyAllWindows()