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
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# Eye functions parameters
WINK_AR_DIFF_THRESH = 0.04
WINK_AR_CLOSE_THRESH = 0.19
WINK_CONSECUTIVE_FRAMES = 10
WINK_COUNTER = 0
NEXT_EYE_COUNTER = 0
EYE_AR_THRESH = 0.19
EYE_AR_CONSECUTIVE_FRAMES = 5
EYE_DOWN_CONSECUTIVE_FRAMES = 6
LOOKDOWN_COUNTER = 0
NEXT_EYE_AR_CONSECUTIVE_FRAMES = 40
countt = 0
FIRST_EYE = False
MOUTH_AR_THRESH = 0.3
mouseMovement = False
MOUTH_CONSECUTIVE_FRAME = 10
MOUTH_COUNTER = 0
EYE_DOWN = False
EYE_UP = False


# Eye position details
previous_area = None
previous_keypoints = None
previous_x = None
previous_y = None


ss = ''

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
    
    # print(keypoints)
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

        # print(ans)
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


def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    # print(mouth[13],mouth[14])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return mar


# calculation
def get_gaze_ratio(eye_points, facial_landmarks,frame,prevArea,prevKeypoint,prevX,prevY,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss,mouseMovement,MOUTH_COUNTER,EYE_DOWN,EYE_UP):


    shape = face_utils.shape_to_np(facial_landmarks)
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    mouthEAR = mouth_aspect_ratio(mouth)

    diff_ear = np.abs(leftEAR - rightEAR)

    leftEARPER = 100*leftEAR
    


    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    nose_point = (landmarks.part(27).x, landmarks.part(27).y)
    last_point = (landmarks.part(8).x, landmarks.part(8).y)
    
    if diff_ear<WINK_AR_DIFF_THRESH:
        LOOKDOWN_COUNTER += 1
    else:
        LOOKDOWN_COUNTER = 0



    height, width, _ = frame.shape

    min_x = np.min(left_eye_region[:, 0]) + 7
    max_x = np.max(left_eye_region[:, 0]) 
    min_y = np.min(left_eye_region[:, 1]) - 6
    max_y = np.max(left_eye_region[:, 1]) + 7

    roi = frame[min_y: max_y,min_x: max_x]

    TH = cv2.getTrackbarPos("TH","Trackbar")
    keypoints = process_eye(roi,TH,blobl_detector,prevArea)
    # print(keypoints[0].size)
    # print(prevKeypoint)



    new_x = 0
    new_y = 0
    if keypoints:
        prevKeypoint = keypoints
        prevArea = keypoints[0].size
        xx = keypoints[0].pt[0]
        yy = keypoints[0].pt[1]
        prevX = xx
        prevY = yy
        new_x = prevX
        new_y = prevY 

    else:
        keypoints = prevKeypoint
        new_x = prevX
        new_y = prevY 


    ear = (leftEAR+rightEAR)/2.0
    
    leftCLCK = False
    rightCLCK = False
    if ear<= EYE_AR_THRESH:
        WINK_COUNTER += 1
        
    else :
        if WINK_COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
            countt += 1

        WINK_COUNTER = 0
        if countt > 0:
            NEXT_EYE_COUNTER += 1
            if NEXT_EYE_COUNTER > NEXT_EYE_AR_CONSECUTIVE_FRAMES:
                if countt == 1:
                    print("right")
                    pag.click(button='right')
                elif countt == 2:
                    print("left")
                    pag.click(button='left')
                elif countt == 3:
                    print("up")
                    pag.moveRel(0, 5)
                    EYE_UP = True
                else:
                    print("down")
                    EYE_DOWN = True

                countt = 0
                NEXT_EYE_COUNTER = 0

    cv2.putText(frame, "WINK COUNTER : {:.2f}".format(countt), (300, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if mouthEAR>=MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1
    else:
        if MOUTH_COUNTER>=MOUTH_CONSECUTIVE_FRAME:
            if mouseMovement == False:
                mouseMovement = True
            else:
                mouseMovement = False

        MOUTH_COUNTER = 0 

    # print(mouth)
    #
    # if EYE_DOWN == True and EYE_UP == False:
    #
    #     print("true")
    #     EYE_DOWN = False
    # elif EYE_DOWN == False and EYE_UP == True and ear < 0.25 and ear > 0.17:
    #     pag.moveRel(0, -5)
    #     EYE_UP = False
    # else:
    #     pass

    if new_x == None:
        pass
    else :
        poss = np.array([min_x + int(new_x), min_y + int(new_y)])

        diffs = np.linalg.norm(last_point - poss)

        if mouseMovement == True:
            # print(min_x+int(new_x),min_y+int(new_y))
            pos = np.array([min_x+int(new_x),min_y+int(new_y)])
            # print(pos,nose_point)
            dist = np.linalg.norm(nose_point - pos)
            # print(dist)




            cv2.circle(frame,(min_x+int(new_x),min_y+int(new_y)),1,(0,255,255),-1)

            direction = None

            if EYE_UP == True:
                if ear < 0.25:
                    pag.moveRel(0, -5)
                else:
                    EYE_UP = False
            elif EYE_DOWN == True:
                if ear < 0.25:
                    pag.moveRel(0, 5)
                else:
                    EYE_DOWN = False
            else:
                if dist < 38:
                    direction = 'left'
                    pag.moveRel(-5, 0)
                elif dist > 38 and dist < 46:
                    direction = 'center'
                    if leftEARPER < 26 and LOOKDOWN_COUNTER > EYE_DOWN_CONSECUTIVE_FRAMES:
                        # pag.moveRel(0, 15)
                        print("ok")
                else:
                    direction = 'right'
                    pag.moveRel(5, 0)

                cv2.putText(frame, "Eyes: " + direction + " {:.2f}".format(dist), (250, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.putText(frame, "diff: {:.2f}".format(diffs), (250, 360),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    roi = cv2.drawKeypoints(roi, keypoints, roi, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.circle(frame, nose_point, 2, (0,255,255), -1)
    # print(new_y,new_x)
    cv2.imshow("rr",roi)




    return prevArea,prevKeypoint,prevX,prevY,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss,mouseMovement,MOUTH_COUNTER,EYE_DOWN,EYE_UP

# main part
while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        landmarks = predictor(gray, face)
        
        previous_area,previous_keypoints,previous_x,previous_y,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss,mouseMovement,MOUTH_COUNTER,EYE_DOWN,EYE_UP = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks,frame,previous_area,previous_keypoints,
            previous_x, previous_y,WINK_COUNTER,LOOKDOWN_COUNTER,NEXT_EYE_COUNTER,FIRST_EYE,countt,ss,mouseMovement,MOUTH_COUNTER,EYE_DOWN,EYE_UP)
        


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()