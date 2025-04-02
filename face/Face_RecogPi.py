#Program to Detect the Face and Recognise the Person based on the data from face-trainner.yml
import time
import cv2 #For Image processing 
import numpy as np #For converting Images to Numerical array 
import os #To handle directories 
from PIL import Image #Pillow lib for handling images 
import sys
arg1 = sys.argv[1]
# Set the time to run the detection algorithm for
max_time = 30  # 60 seconds

start_time = time.time()

# Get list of filenames in directory and extract labels from filenames
path = "/Users/kostasbekis/live_face_recognition/photos"
labels = []
for filename in os.listdir(path):
	label = os.path.splitext(filename)[0]
	labels.append(label)

recognized = False

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

cap = cv2.VideoCapture(0) #Get vidoe feed from the Camera

recognized = False
while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y + h, x:x + w]

		id_, conf = recognizer.predict(roi_gray)
		if conf >= 80:
			font = cv2.FONT_HERSHEY_SIMPLEX
			if id_ < len(labels):
				name = labels[id_]
			if name == arg1:
				recognized = True
			cv2.putText(img, name, (x, y), font, 1, (0, 0, 255), 2)


		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow('Preview', img)
	if recognized == True:
		print(name)

	# Exit if the user presses the 'q' key
	if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time >= 60:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print(recognized)
