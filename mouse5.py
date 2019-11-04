import cv2
import numpy as np
import dlib
from math import hypot
from imutils import face_utils
import pyautogui as pag


# Camera
cap = cv2.VideoCapture(0)

# DLIB DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# BLOB detetor
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
blobl_detector = cv2.SimpleBlobDetector_create(detector_params)


# Facial points
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# Eye functions parameters
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10
WINK_COUNTER = 0
NEXT_EYE_COUNTER = 0
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 2
EYE_DOWN_CONSECUTIVE_FRAMES = 6
LOOKDOWN_COUNTER = 0
NEXT_EYE_AR_CONSECUTIVE_FRAMES = 15
FIRST_EYE = False
countt = 0


# Eye position details
previous_area = None
previous_keypoints = None
previous_x = None
previous_y = None



# SLIDEBAR
def nothing(x):
    pass


cv2.namedWindow("Trackbar")
cv2.createTrackbar("TH", "Trackbar", 0 , 179, nothing)



# processing part
def process_eye(img, threshold, detector, prevArea=None):
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("img1",img)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    
    
    keypoints = detector.detect(img)
    
    img = cv2.resize(img,(150,100))
    cv2.imshow("img2",img)
    
    print(keypoints)
    x = 0
    y = 0
    if keypoints and prevArea and len(keypoints) > 1:
        tmp = 1000
        for keypoint in keypoints:  # filter out odd blobs
            if abs(keypoint.size - prevArea) < tmp:
                ans = keypoint
                x = keypoint.pt[0]
                y = keypoint.pt[1]
                tmp = abs(keypoint.size - prevArea)

        print(ans)        
        keypoints = [ans]

    # print(keypoints)
    # print(x,y)
    return keypoints


# EAR 
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)

    return ear



# calculation
def get_gaze_ratio(eye_points, facial_landmarks,frame,prevArea,prevKeypoint,prevX,prevY,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss):


    shape = face_utils.shape_to_np(facial_landmarks)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR+rightEAR)/2.0
    # print(ear,leftEAR,rightEAR)

    print(NEXT_EYE_COUNTER,countt,WINK_COUNTER)
    

    leftCLCK = False
    rightCLCK = False
    if ear<= EYE_AR_THRESH:
        WINK_COUNTER += 1
        
    else :
        if WINK_COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
            countt += 1

            if countt == 2 and NEXT_EYE_COUNTER <= NEXT_EYE_AR_CONSECUTIVE_FRAMES:
                countt = 0
                ss = 'left'
                pag.click(button='left')

        WINK_COUNTER = 0
        if countt == 1:
            NEXT_EYE_COUNTER += 1
            if NEXT_EYE_COUNTER > NEXT_EYE_AR_CONSECUTIVE_FRAMES:
                ss = 'right'
                countt = 0
                NEXT_EYE_COUNTER = 0
                pag.click(button='right')

        
    

    cv2.putText(frame, "mouth: {:.2f}".format(countt)+ " {:.2f}".format(WINK_COUNTER)+" {:.2f}".format(NEXT_EYE_COUNTER)+ss, (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, ss, (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    


    return prevArea,prevKeypoint,prevX,prevY,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss
    

ss = ''
# main part
while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
                
        previous_area,previous_keypoints,previous_x,previous_y,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks,frame,previous_area,previous_keypoints,
            previous_x, previous_y,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss) 
        


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()