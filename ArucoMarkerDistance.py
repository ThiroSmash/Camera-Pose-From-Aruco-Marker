# import the necessary packages
from __future__ import print_function
import datetime
from threading import Thread
import argparse
import imutils
import cv2
import cv2.aruco as aruco
import numpy as np



'''
	WebcamVideoStream class
	Simple class to encapsulate threaded usage of OpenCV's video capturing tools
	from PyImageSearch
'''
class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
	def read(self):
		# return the frame most recently read
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", type=int, default=0,
	help="Puerto de cámara a utilizar (para más información, ejecute testports.py)")

args = vars(ap.parse_args())



#create a threaded video stream
print("Initialising webcam and parameters...")
vs = WebcamVideoStream(src=args["port"]).start()

#prerequisites of aruco detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

#read intrinsic camera parameters
matrix = np.loadtxt("intrinsic_parameters.txt", float)
focalLengthX = matrix[0][0]

#Real width of our markers is 6.35 cm
realWidth = 6.35

#Marker corners coordinates
objPoints = np.array([
			(0,0,0),
			(6.35,0,0),
			(6.35,6.35,0),
			(0,6.35,0)
			])

#distortion coefficients
dist_coeffs = np.loadtxt("distortion_coefficients.txt",float)

#Record and search for markers until stop signal is given
while (True):
	# grab the frame from the threaded video stream
	frame = vs.read()

	# detect aruco markers in the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	#if a marker has been detected, calculate distance to it
	if(not(ids==None)):

		#show the frame, with detected markers
		gray = aruco.drawDetectedMarkers(gray, corners)
		imgPoints =  np.array(corners[ids[0][0]])		
		success, rotation_vector, translation_vector = cv2.solvePnP(objPoints, imgPoints, matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

		if(success):		
			#show coordinates in image output
			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = [30,30]
			fontScale              = 0.7
			fontColor              = (255,255,255)
			lineType               = 2
			text = "X: " + str(translation_vector[0])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Y: " + str(translation_vector[1])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Z: " + str(translation_vector[2])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)


	cv2.imshow("Frame", gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()