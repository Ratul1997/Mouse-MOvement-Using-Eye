import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def get_gaze_ratio(eye_points, facial_landmarks):
    print("ok")
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # print(left_eye_region[:, 0])
    # print(left_eye_region[:, 1])
    # print(left_eye_region)


    ret, thresh_gray = cv2.threshold(eye, 73, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gauss = cv2.adaptiveThreshold(thresh_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)

    # cv2.drawContours(eye,contours,-1,(0,0,255),6)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])


    roi = frame[min_y: max_y, min_x: max_x]
    roi = cv2.resize(roi,(150,100))
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    print(left_side_white)

    roi_gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(roi_gray)
    thres=cv2.inRange(equ,0,20)
    kernel = np.ones((3,3),np.uint8)
    
    dilation = cv2.dilate(thres,kernel,iterations = 2)
    
    erosion = cv2.erode(dilation,kernel,iterations = 3)
    
    contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    print(len(contours))

    if left_side_white > 100:
        cv2.putText(frame, str(left_side_white), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        new_frame[:] = (0, 0, 255)
        # cv2.imshow("new",new_frame)
    else : 
        
        new_frame[:] = (0,0,0)
    

    print(right_side_white)


    cv2.imshow("new", thresh_gray)
    cv2.imshow("roi", roi)



while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2  .COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)




    cv2.imshow("Frame", frame)
    # cv2.imshow("New frame", new_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()