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
	def __init__(self, port=0, maxSuccesses=10, raw=False, showOriginals=False, inverseX=True, inverseY=False, inverseZ=False, inverseXAngle=False, inverseYAngle=True, inverseZAngle=True, inverseXMarker=False, inverseYMarker=True, inverseZMarker=True):

		#create a threaded video stream
		self.vs = WebcamVideoStream(src=port).start()

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
		mtx = np.loadtxt("marker_points.txt")
		self.markerPoints = np.array(mtx[:,0:3])

		self.inverseXMarker = inverseXMarker
		if(inverseXMarker):
			self.markerPoints[:,0] = -self.markerPoints[:,0]

		self.inverseYMarker = inverseYMarker
		if(inverseYMarker):
			self.markerPoints[:,1] = -self.markerPoints[:,1]

		self.inverseZMarker = inverseZMarker
		if(inverseZMarker):
			self.markerPoints[:,2] = -self.markerPoints[:,2]

		print(self.markerPoints)

		self.markerIds = mtx[::4,3]
		self.maxSuccesses = maxSuccesses
		self.inverseX = inverseX
		self.inverseY = inverseY
		self.inverseZ = inverseZ
		self.inverseXAngle = inverseXAngle
		self.inverseYAngle = inverseYAngle
		self.inverseZAngle = inverseZAngle
		self.outputRaw = raw
		self.showOriginals = showOriginals

	def processFrame(self):
		# grab the frame from the threaded video stream
		while(True):
			frame, repeated = self.vs.read()
			if(not repeated):
				break

		# detect aruco markers in the frame
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		corners, idsM, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters, cameraMatrix=self.refinedMatrix, distCoeff=self.dist_coeffs)

		#remove detected markers that aren't defined in marker_points.txt, and change format to array from matrix
		ids = []
		if(idsM is not None):
			#change ids format to array
			for i in idsM:
				ids = np.append(ids, i[0])
			#remove undefined ids
			for i in ids:
				if(i not in self.markerIds):
					pos = np.where(ids == i)[0][0] #the reason for these brackets is to undo numpy's return of array of arrays
					ids = np.delete(ids, pos)

		success = False
		world_coordinates = []
		rot_vector = []
		trans_vector = []
		retIds = []

		#if a defined marker has been detected
		if(len(ids) > 0):

			#create objPoints depending on which markers were detected, respecting
			#the order of 'corners' and 'ids'
			objPoints = np.empty((0,3),float)
			imgPoints = np.empty((0,2), float)
			for i in ids:
				#self.markerIds may be unordered, so we must find the index of each id
				pos = np.where(self.markerIds == i)[0][0]
				#extract the corresponding points of the id
				markerIPoints = self.markerPoints[pos*4:(pos+1)*4][:]
				#insert the matrix into our final objPoints
				objPoints = np.vstack((objPoints, markerIPoints))
				#insert the corresponding detected corners into imgPoints
				posCorner = np.where(idsM == i)[0][0]
				imgPoints = np.vstack((imgPoints, corners[posCorner][0]))
				retIds.append(i)

			#show the frame, with detected markers
			gray = aruco.drawDetectedMarkers(gray, corners)

			success, rotation_vector, translation_vector = cv2.solvePnP(objPoints, imgPoints, self.refinedMatrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
			translation_vector.mean()
			if(success):
				#solvePnP's x-axis rotation angle is of opposite sign relative to the Y coordinate
				rotation_vector[0] = -rotation_vector[0]
				world_coordinates = self.camera_to_world_coords(rotation_vector, translation_vector)

				if(self.inverseX):
					world_coordinates[0] = -world_coordinates[0]

				if(self.inverseY):
					world_coordinates[1] = -world_coordinates[1]

				if(self.inverseZ):
					world_coordinates[2] = -world_coordinates[2]

				#turn rotation_vector from radians to degrees
				rotation_vector = rotation_vector / math.pi * 180

				if(self.inverseXAngle):
					rotation_vector[0] = -rotation_vector[0]

				if(self.inverseYAngle):
					rotation_vector[1] = -rotation_vector[1]

				if(self.inverseZAngle):
					rotation_vector[2] = -rotation_vector[2]

				#build output arrays
				for i in range(3):
					rot_vector.append((rotation_vector[i]).item())
					trans_vector.append((translation_vector[i]).item())

		return success, world_coordinates, rot_vector, trans_vector, retIds, gray

	def video(self):
		while (True):
			success, world_coordinates, rotation_vector, translation_vector, detectedMarkers, gray = self.processFrame()
			#show data in output image
			if(success):

				#Round outputs to 3 decimals
				for i in range(3):
					world_coordinates[i] = round(world_coordinates[i], 3)
					rotation_vector[i] = round(rotation_vector[i], 3)
					translation_vector[i] = round(translation_vector[i], 3)
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

				if(self.showOriginals):
					#show original coordinates from OpenCV
					bottomLeftCornerOfText = [300, 30]
					text = "Original coords:"
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
		realCoordinatesMatrix = np.loadtxt("camera_points.txt")
		assert len(realCoordinatesMatrix[0]) == 6
		meanErrorsArray = []
		rawErrorsArray = np.empty((0,6), float)
		nPoints = len(realCoordinatesMatrix)
		finalDetectedMarkers = []
		for i in range(nPoints):
			successes = 0
			errors_i = []
			print("")
			print("Next pose:")
			print("X: " + str(realCoordinatesMatrix[i][0]))
			print("Y: " + str(realCoordinatesMatrix[i][1]))
			print("Z: " + str(realCoordinatesMatrix[i][2]))
			print("X-axis angle: " + str(realCoordinatesMatrix[i][3]))
			print("Y-axis angle: " + str(realCoordinatesMatrix[i][4]))
			print("Z-axis angle: " + str(realCoordinatesMatrix[i][5]))
			print("")
			input("Position camera and press Enter to continue.")
			print("")
			#take multiple shots of the point in question. if marker can't be detected, notify user
			failures = 0
			failed = False
			while(successes < self.maxSuccesses):
				success, world_coordinates, rotation_vector, translation_vector, detectedMarkers, frame = self.processFrame()
				if(success):
					failures = 0
					calculatedCoordinates = world_coordinates + rotation_vector
					successes += 1
					error = realCoordinatesMatrix[i] - calculatedCoordinates
					errors_i.append(error)
					#Check and store whether new markers have been detected
					for id in detectedMarkers:
						if id not in finalDetectedMarkers:
							finalDetectedMarkers.append(id)

				else:
					failures = failures + 1
				if(failures >= 100):
					inp = input("Could not detect a marker from this pose. Would you like to try again? [y/n]:")
					print("")
					if(inp == 'n'):
						print("Skipping current pose. Result will be filled with '-' for this pose.")
						failed = True
						break
					else:
						if(inp == 'y'):
							failures = 0
							input("Reposition camera and press Enter to continue.")
							print("")
						else:
							print("Sorry, could not recognise answer. Trying one more sample...")
							print("")

			#apply filters to error results
			if(not failed):

				#add raw errors to results
				for i in range(self.maxSuccesses):
					for j in range(6):
						errors_i[i][j] = round(errors_i[i][j], 3)

				rawErrorsArray = np.vstack((rawErrorsArray, errors_i))

				#calculate mean of each coordinate from all shots taken
				meanError = np.mean(errors_i, axis=0)

				for i in range(6):
					meanError[i] = round(meanError[i], 3)

				#add result to errors array
				meanErrorsArray.append(meanError)
			else:
				failedArray = ['-','-','-','-','-','-']

				for i in range(self.maxSuccesses):
					rawErrorsArray.append(failedArray)

				meanErrorsArray.append(failedArray)


		#save results in txt

		file = open("results.txt", "a")

		#save outputs with context and filters
		if(not self.outputRaw):


			file.write("\nCoordinate systems:\n")
			file.write("Markers: ")
			if(self.inverseXMarker):
				file.write("X-left, ")
			else: file.write("X-right, ")
			if(self.inverseYMarker):
				file.write("Y-up, ")
			else: file.write("Y-down, ")
			if(self.inverseZMarker):
				file.write("Z-backward")
			else: file.write("Z-forward")
			file.write("\nCamera coordinates: ")
			if(self.inverseX):
				file.write("X-right, ")
			else: file.write("X-left, ")
			if(self.inverseY):
				file.write("Y-down, ")
			else: file.write("Y-up, ")
			if(self.inverseZ):
				file.write("Z-forward")
			else: file.write("Z-backward")
			file.write("\nCamera angles: ")
			if(self.inverseXAngle):
				file.write("X-down, ")
			else: file.write("X-up, ")
			if(self.inverseYAngle):
				file.write("Y-right, ")
			else: file.write("Y-left, ")
			if(self.inverseZAngle):
				file.write("Z-clockwise\n")
			else: file.write("Z-counterclockwise\n")

			file.write("\nMarker definitions:\n")

			mtx = np.loadtxt("marker_points.txt")
			markersList = np.array(mtx[:,0:3]).tolist()

			for i in range(len(markersList)):
				strPoint = [str(point) for point in markersList[i]]
				joinPoints = " ".join(strPoint)
				file.write(str(int(self.markerIds[math.trunc(i/4)])) + ": ")
				file.writelines(joinPoints)
				file.write("\n")

			file.write("\nFinal detected markers:\n")

			strId = [str(int(id)) for id in finalDetectedMarkers]
			joinId = " ".join(strId)
			file.writelines(joinId)
			file.write("\n")

			file.write("\nPose definitions:\n")

			posesList = realCoordinatesMatrix.tolist()

			for i in range(len(posesList)):
				strPoint = [str(point) for point in posesList[i]]
				joinPoints = " ".join(strPoint)
				file.write(str(i+1) + ": ")
				file.writelines(joinPoints)
				file.write("\n")

			file.write("\nEstimation errors:\n")

			file.write("\nWith mean filter:\n")

			for i in range(nPoints):
				strPoint = [str(point) for point in meanErrorsArray[i]]
				joinPoints = " ".join(strPoint)

				file.write(str(i+1) + ": ")
				file.writelines(joinPoints)
				file.write("\n")

			file.write("\nRaw errors:\n")

			for i in range(nPoints):
				for j in range(self.maxSuccesses):

					strPoint = [str(point) for point in rawErrorsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.write(str(i+1) + "." + str(j+1) + ": ")
					file.writelines(joinPoints)
					file.write("\n")

	    #save only raw outputs
		else:
			for i in range(nPoints):
				for j in range(self.maxSuccesses):
					strPoint = [str(point) for point in rawErrorsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.writelines(joinPoints)
					file.write("\n")

		file.close()
		print("")
		print("Estimation errors successfully saved in results.txt")

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
ap.add_argument("-p", "--Port", type=int, default=0,
	help="Camera port to use (for more info, run testports.py), port 0 by default")

ap.add_argument("-x", "--MaxSuccesses", type=int, default=10,
	help="Number of required successful frames for each shot in snapshot mode, 10 by default")

ap.add_argument("-iX", "--InverseX", default=True, action='store_false',
	help="Inverts X coordinate output. (Default: positive X-right, Y-up, Z-backwards)")

ap.add_argument("-iY", "--InverseY", default=False, action='store_true',
	help="Inverts Y coordinate output. (Default: positive X-right, Y-up, Z-backwards)")

ap.add_argument("-iZ", "--InverseZ", default=False, action='store_true',
	help="Inverts Z coordinate output. (Default: positive X-right, Y-up, Z-backwards)")

ap.add_argument("-iXA", "--InverseXAngle", default=False, action='store_true',
	help="Inverts X-axis angle output. (Default: positive X-up, Y-right, Z-clockwise)")

ap.add_argument("-iYA", "--InverseYAngle", default=True, action='store_false',
	help="Inverts Y-axis angle output. (Default: positive X-up, Y-right, Z-clockwise)")

ap.add_argument("-iZA", "--InverseZAngle", default=True, action='store_false',
	help="Inverts Z-axis angle output. (Default: positive X-up, Y-right, Z-clockwise)")

ap.add_argument("-iXM", "--InverseXMarker", default=False, action='store_true',
	help="Inverts X coordinate of marker inputs. (Default same coordinate system as default output)")

ap.add_argument("-iYM", "--InverseYMarker", default=True, action='store_false',
	help="Inverts X coordinate of marker inputs. (Default same coordinate system as default output)")

ap.add_argument("-iZM", "--InverseZMarker", default=True, action='store_false',
	help="Inverts X coordinate of marker inputs. (Default same coordinate system as default output)")

ap.add_argument("-s", "--Snapshot", default=False, action='store_true',
	help="Turns on snapshot mode (requires camera_points.txt, outputs estimation errors in results.txt)")

ap.add_argument("-r", "--Raw", default=False, action='store_true',
	help="Snapshot mode outputs raw error arrays without context data or filters (only applicable in snapshot mode)")

ap.add_argument("-o", "--ShowOriginals", default=False, action='store_true',
	help="Shows coordinates directly calculated by OpenCV (relative to camera's orientation) (only applicable in video mode)")

args = vars(ap.parse_args())

pd = PoseDetector(args['Port'], args['MaxSuccesses'],  args['Raw'], args['ShowOriginals'], args['InverseX'], args['InverseY'], args['InverseZ'], args['InverseXAngle'], args['InverseYAngle'], args['InverseZAngle'], args['InverseXMarker'], args['InverseYMarker'], args['InverseZMarker'])

if(not args['Snapshot']):
	pd.video()
else:
	pd.snapshots()

print("Closing software...")
pd.stop()
cv2.destroyAllWindows()
