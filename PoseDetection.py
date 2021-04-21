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
		self.repeated = False
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
			self.repeated = False

	def read(self):
		# return the frame most recently read
		if(self.repeated):
			return self.frame, self.repeated
		else:
			self.repeated = True
			return self.frame, not self.repeated
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True






class PoseDetector:
	def __init__(self, port=0, maxSuccesses=10):

		#create a threaded video stream
		self.vs = WebcamVideoStream(src=args["port"]).start()

		#prerequisites of aruco detection
		self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
		self.parameters =  aruco.DetectorParameters_create()
		self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

		#read intrinsic camera parameters
		self.matrix = np.loadtxt("camera_matrix.txt", float)
		self.refinedMatrix = np.loadtxt("refined_camera_matrix.txt", float)
		self.focalLengthX = self.matrix[0][0]
		self.dist_coeffs = np.loadtxt("distortion_coefficients.txt",float)
		#Marker corners coordinates
		mtx = np.loadtxt("markerPoints.txt")
		self.objPoints = np.array(mtx[:,0:3])
		self.ids = mtx[::4,3]
		self.maxSuccesses = maxSuccesses

	def processFrame(self):
		# grab the frame from the threaded video stream
		while(True):
			frame, repeated = self.vs.read()
			if(not repeated):
				break

		# detect aruco markers in the frame
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters, cameraMatrix=self.refinedMatrix, distCoeff=self.dist_coeffs)

		success = False
		world_coordinates = []
		rot_vector = []
		trans_vector = []

		#if a marker has been detected
		if(not(ids==None)):

			#show the frame, with detected markers
			gray = aruco.drawDetectedMarkers(gray, corners)
			imgPoints =  np.array(corners[ids[0][0]])
			success, rotation_vector, translation_vector = cv2.solvePnP(self.objPoints, imgPoints, self.refinedMatrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
			translation_vector.mean()
			if(success):
				rotation_vector[0] = -rotation_vector[0]
				world_coordinates = self.camera_to_world_coords(rotation_vector, translation_vector)
				#turn rotation_vector from radians to degrees
				rotation_vector = rotation_vector / math.pi * 180

				#build output arrays
				for i in range(3):
					rot_vector.append((rotation_vector[i]).item())
					trans_vector.append((translation_vector[i]).item())

		return success, world_coordinates, rot_vector, trans_vector, gray



	def video(self):
		while (True):
			success, world_coordinates, rotation_vector, translation_vector, gray = self.processFrame()
			print(world_coordinates[0])
			#show data in output image
			if(success):
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

				#show angles
				text = "Rotation (degrees):"
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "X: " + str(rotation_vector[0])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "Y: " + str(rotation_vector[1])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "Z: " + str(rotation_vector[2])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)

				#show coordinates relative to rotation
				bottomLeftCornerOfText = [300, 30]
				text = "Relative coords:"
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "X: " + str(translation_vector[0])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "Y: " + str(translation_vector[1])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30
				text = "Z: " + str(translation_vector[2])
				cv2.putText(gray, text, tuple(bottomLeftCornerOfText), font, fontScale, fontColor, lineType)
				bottomLeftCornerOfText[1] += 30

			cv2.imshow("Frame", gray)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

	def snapshots(self):
		realCoordinatesArray = np.loadtxt("cameraPoints.txt")
		assert len(realCoordinatesArray[0]) == 6
		errorsArray = []
		nPoints = len(realCoordinatesArray)

		for i in range(nPoints):
			successes = 0
			errors_i = []
			input("Press Enter to continue...")
			#take multiple shots of the point in question
			while(successes < self.maxSuccesses):
				success, world_coordinates, rotation_vector, translation_vector, frame = self.processFrame()
				if(success):
					calculatedCoordinate = world_coordinates + rotation_vector
					successes += 1
					error = realCoordinatesArray[i] - calculatedCoordinate
					errors_i.append(error)

			#calculate mean of each coordinate from all shots taken
			meanError = np.mean(errors_i, axis=0)
			#add result to errors array
			errorsArray.append(meanError)

		#save results in txt
		np.savetxt("results.txt", errorsArray)

	def stop(self):
		self.vs.stop()

	def camera_to_world_coords(self, rotation_vector, translation_vector):
		world_coordinates = [0,0,0]

		world_coordinates[0] = ( translation_vector[0]*math.cos(rotation_vector[1]) - translation_vector[2]*math.sin(rotation_vector[1]) ).item()
		world_coordinates[1] = ( translation_vector[1]*math.cos(rotation_vector[0]) - translation_vector[2]*math.sin(rotation_vector[0]) ).item()
		world_coordinates[2] = ( translation_vector[2]*math.cos(rotation_vector[1])*math.cos(rotation_vector[0]) + translation_vector[0]*math.sin(rotation_vector[1]) + translation_vector[1]*math.sin(rotation_vector[0]) ).item()

		return world_coordinates


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", type=int, default=0,
	help="Camera port to use (for more info, run testports.py)")
ap.add_argument("-m", "--mode", type=int, default=0,
	help="Mode to execute the program: 0 = video mode (outputs video with coordinates), 1 = snapshots mode (requires cameraPoints.txt, outputs errors in results.txt)")
ap.add_argument("-x", "--maxSuccesses", type=int, default=10,
	help="Number of required successful frames for each shot in snapshot mode")

args = vars(ap.parse_args())

pd = PoseDetector(args['port'], args['maxSuccesses'])

if(args['mode']==0):
	pd.video()
else:
	if(args['mode']==1):
		pd.snapshots()
pd.stop()
print("Ending...")
# cleanup
cv2.destroyAllWindows()
