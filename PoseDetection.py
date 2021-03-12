# import the necessary packages
from __future__ import print_function
import datetime
from threading import Thread
import argparse
import imutils
import cv2
import cv2.aruco as aruco
import numpy as np
import math


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
'''
	camera_to_world_coords function
	Function to convert solvePnP's translation vector from relative to the camera's rotation to relative to the marker's rotation
'''
def camera_to_world_coords(rotation_vector, translation_vector):
	world_coordinates = [0,0,0]
	world_coordinates[0] = translation_vector[0]*math.cos(rotation_vector[1]) - translation_vector[2]*math.sin(rotation_vector[1])
	world_coordinates[1] = translation_vector[1]*math.cos(rotation_vector[0]) - translation_vector[2]*math.sin(rotation_vector[0])
	world_coordinates[2] = translation_vector[2]*math.cos(rotation_vector[1])*math.cos(rotation_vector[0]) + translation_vector[0]*math.sin(rotation_vector[1]) + translation_vector[1]*math.sin(rotation_vector[0])
	return world_coordinates

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", type=int, default=0,
	help="Camera port to use (for more info, run testports.py)")

args = vars(ap.parse_args())

#create a threaded video stream
print("Initialising webcam and parameters...")
vs = WebcamVideoStream(src=args["port"]).start()

#prerequisites of aruco detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

#read intrinsic camera parameters
matrix = np.loadtxt("camera_matrix.txt", float)
refinedMatrix = np.loadtxt("refined_camera_matrix.txt", float)
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
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=refinedMatrix, distCoeff=dist_coeffs)

	#if a marker has been detected after undistortion
	if(not(ids==None)):
		
		#show the frame, with detected markers
		gray = aruco.drawDetectedMarkers(gray, corners)
		imgPoints =  np.array(corners[ids[0][0]])		
		success, rotation_vector, translation_vector = cv2.solvePnP(objPoints, imgPoints, refinedMatrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

		if(success):
			
			world_coordinates = camera_to_world_coords(rotation_vector, translation_vector)
				
			#show coordinates in image output
			font                   = cv2.FONT_HERSHEY_SIMPLEX
			bottomLeftCornerOfText = [30,30]
			fontScale              = 0.7
			fontColor              = (255,255,255)
			lineType               = 2
			text = "Coordinates (cm):"
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "X: " + str(world_coordinates[0])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Y: " + str(world_coordinates[1])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Z: " + str(world_coordinates[2])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 40

			#turn rotation_vector from radians to degrees
			rotation_vector_degrees = rotation_vector / math.pi * 180
			
			#show angles
			text = "Rotation (degrees):"
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "X: " + str(rotation_vector_degrees[0])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Y: " + str(rotation_vector_degrees[1])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30
			text = "Z: " + str(rotation_vector_degrees[2])
			cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
			bottomLeftCornerOfText[1] += 30


	cv2.imshow("Frame", gray)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# cleanup
cv2.destroyAllWindows()
vs.stop()