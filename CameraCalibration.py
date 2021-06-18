from __future__ import print_function
import datetime
from threading import Thread
import argparse
import imutils
import cv2.aruco as aruco
import numpy as np
import cv2
import time


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
	help="Camera port (for more info, run testports.py)")
ap.add_argument("-i", "--iterations", type=int, default=10,
	help="Amount of valid samples to require for the calibration")

ap.add_argument("-crl", "--CropLeft", type=int, default=0,
	help="Crop images from the left by amount of pixels, zero by default.")

ap.add_argument("-crr", "--CropRight", type=int, default=0,
	help="Crop images from the right by amount of pixels, zero by default.")

ap.add_argument("-crt", "--CropTop", type=int, default=0,
	help="Crop images from the top by amount of pixels, zero by default.")

ap.add_argument("-crb", "--CropBottom", type=int, default=0,
	help="Crop images from the bottom by amount of pixels, zero by default.")

args = vars(ap.parse_args())

cropTop = args['CropTop']
cropBottom = args['CropBottom']
cropLeft = args['CropLeft']
cropRight = args['CropRight']
#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) * 0.79375 #real size in cm of the squares that we'll use

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#image size parameters
height = 0
width = 0

# create a threaded video stream
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=args["port"]).start()

# loop over frames until all samples are taken
nFound = 0
print("Grabbing chessboard samples from video stream.")
lastSampleTime = -3
reprojSample = None
while nFound < args["iterations"]:
	if not (lastSampleTime + 3 > time.time()):	# Calling time.time() every frame is highly
							# ineffficient. However, time.sleep() messes up cv2's imshow
							# function and efficiency is not of concern for this program
		# grab the frame from the threaded video stream
		frame = vs.read()

		heightI, widthI, trash = frame.shape

		croppedFrame = frame[cropTop:(heightI-cropBottom), cropLeft:(widthI-cropRight)].copy()

		cv2.imshow('img', croppedFrame)
		gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)
		print(gray.shape)
		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
		# If found, add object points, image points (after refining them)
		if ret == True:
			nFound = nFound + 1
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners)
			# Draw and display the corners
			cv2.drawChessboardCorners(croppedFrame, (7,6), corners2, ret)
			showFrame = imutils.resize(croppedFrame, width=600)
			cv2.imshow('img', showFrame)
			if nFound < args["iterations"]:
				print("Sample " + str(nFound) + " saved successfully! 3 seconds until next sample...")
				lastSampleTime = time.time()

			else:
				height, width = croppedFrame.shape[:2]
				reprojSample = croppedFrame
				print("Last sample saved successfully!")

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
if(ret):
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (width,height), 1, (width,height))
	print("Parameters Matrix:")
	print(mtx)
	print("Distortion coefficients:")
	print(dist_coeffs)
	np.savetxt("camera_matrix.txt", mtx)
	np.savetxt("refined_camera_matrix.txt", newcameramtx)
	np.savetxt("distortion_coefficients.txt", dist_coeffs)

	#Error calculation of the estimated parameters
	#Undistortion
	dst = cv2.undistort(reprojSample, mtx, dist_coeffs, None, newcameramtx)

	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imshow('Undistorted sample', dst)

	#Re-projection error
	meanError = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist_coeffs)
		error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		meanError += error
	print("Total error: {}".format(meanError/len(objpoints)))

else:
	print("Error calculating parameters.")

print("Press Q to exit")
while not( cv2.waitKey(1) & 0xFF == ord('q') ):
	a = 0
cv2.destroyAllWindows()
vs.stop()
