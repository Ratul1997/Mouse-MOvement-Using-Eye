from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])

    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)

    return mar

def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	return ear
 
EYE_AR_THRESH = 0.3
MOUTH_AR_THRESH = 0.8
EYE_AR_CONSEC_FRAMES = 3
DOWZY_CONSEC_FRAMES = 40

# initialize the frame YAWN_COUNTERs and the total number of blinks
YAWN_COUNTER = 0
DAWZY_COUNER = 0 
TOTAL = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print(lStart,lEnd)

cap = cv2.VideoCapture(0) 
writer = None


while True:

	ret,frame = cap.read()
	frame = cv2.resize(frame, (450,450))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		mouth = shape[mStart:mEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		mouthEAR = mouth_aspect_ratio(mouth)

		print(leftEAR,rightEAR)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		diff_ear = np.abs(leftEAR - rightEAR)

		

		cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		cv2.putText(frame, "Rigth EAR: {:.2f}".format(rightEAR), (50, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		 
	# show the frame


	# if the writer is not None, write the frame with recognized
	# faces t odisk)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.waitKey(0)
cv2.destroyAllWindows()

