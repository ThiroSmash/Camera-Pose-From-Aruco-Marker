# import the necessary packages
from __future__ import print_function
import datetime
from threading import Thread
#from imutils.video import WebcamVideoStream
#from imutils.video import FPS
import argparse
import imutils
import cv2
import cv2.aruco as aruco

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

#prerequisites of aruco detection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

#create a threaded video stream
print("Initialising webcam and parameters...")
vs = WebcamVideoStream(src=args["port"]).start()

#Record and search for markers until stop signal is given
while (True): #args["num_frames"]: #ending condition, should maybe change to a key input
	# grab the frame from the threaded video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# our aruco operations on the frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	print(corners)

	#show the frame with the detected markers
	gray = aruco.drawDetectedMarkers(gray, corners)
	cv2.imshow("Frame", gray)
	# update the FPS counter
	fps.update()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()