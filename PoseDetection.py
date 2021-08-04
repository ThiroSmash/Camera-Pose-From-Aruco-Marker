# import the necessary packages
from __future__ import print_function
import datetime
from threading import Thread
import argparse
import imutils
import cv2
import cv2.aruco as aruco
from PIL import ImageGrab
import numpy as np
import math
import sys


'''
	WebcamVideoStream class
	Simple class to encapsulate threaded usage of OpenCV/PIL's video capturing tools
	from PyImageSearch with slight modifications
'''
class WebcamVideoStream:
	def __init__(self, src=0, bbox=-1):
		# initialize the video camera stream and read the first frame
		# from the stream
		if(bbox == -1):
			self.stream = cv2.VideoCapture(src)
			self.recordScreen = False
			(self.grabbed, self.frame) = self.stream.read()
		else:
			self.bbox = tuple(bbox)
			self.recordScreen = True
			self.frame = ImageGrab.grab(self.bbox)
			self.frame = cv2.cvtColor(np.array(self.frame), cv2.COLOR_RGB2BGR)
		self.repeated = False

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
			if(self.recordScreen):
				self.frame = ImageGrab.grab(bbox=self.bbox)
				#self.frame = cv2.cvtColor(np.array(self.frame), cv2.COLOR_RGB2BGR)
			else:
				(self.grabbed, self.frame) = self.stream.read()
			self.repeated = False

	def read(self):
		# return the frame most recently read
		if(self.recordScreen):
			self.frame = cv2.cvtColor(np.array(self.frame), cv2.COLOR_RGB2BGR)
		if(self.repeated):
			return self.frame, self.repeated
		else:
			self.repeated = True
			return self.frame, not self.repeated
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


class PoseDetector:

	SIMPLE_MOVING_AVERAGE = 1
	WEIGHTED_MOVING_AVERAGE = 2
	EXPONENTIAL_MOVING_AVERAGE = 3
	MOVING_MEDIAN = 4

	def __init__(self, port=0, boundingBox=-1, showOriginals=False, defaultCalibration=False, applyDisplacement=False):

		#create a threaded video stream
		self.vs = WebcamVideoStream(src=port, bbox=boundingBox).start()

		#prerequisites of aruco detection
		self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
		self.parameters =  aruco.DetectorParameters_create()
		self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

		#read intrinsic camera parameters
		if(defaultCalibration):
			self.matrix = np.ones((3,3), dtype=float)
			self.matrix[0,1] = 0
			self.matrix[1,0] = 0
			self.matrix[2,0:2] = 0
			self.refinedMatrix = self.matrix
			self.focalLengthX = self.matrix[0][0]
			self.distCoeffs = np.zeros((5), dtype=float)
		else:

			try:
				self.matrix = np.loadtxt("camera_matrix.txt", float)
				self.refinedMatrix = np.loadtxt("refined_camera_matrix.txt", float)
				self.focalLengthX = self.matrix[0][0]
				self.distCoeffs = np.loadtxt("distortion_coefficients.txt",float)
				assert(self.matrix.shape == (3,3))
				assert(self.refinedMatrix.shape == (3,3))
				assert(self.distCoeffs.shape == (5,))
			except:
				print("Unexpected ", sys.exc_info()[0], "occurred while loading calibration files.")
				print("One or more of the calibration files (camera_matrix.txt, refined_camera_matrix.txt, " +
					"distortion_coefficients.txt) is missing or invalid:")
				sys.exit()

		#Marker corners coordinates
		try:
			msg = "marker_points.txt is missing or invalid."
			mtx = np.loadtxt("marker_points.txt")
			checked, msg = self.__checkMarkerPoints(mtx)
			assert(checked)
		except:
			print("Unexpected ", sys.exc_info()[0], "occurred while loading marker_points.txt:")
			print(msg)
			sys.exit()

		self.markerPoints = np.array(mtx[:,0:3])
		self.markerIds = mtx[::4,3]
		self.showOriginals = showOriginals
		self.applyDisplacement = applyDisplacement

		if(applyDisplacement):
			try:
				self.displacementArray = np.loadtxt("displacement.txt", float)
				assert(self.displacementArray.shape == (3,))
			except:
				print("Unexpected ", sys.exc_info()[0], "occurred while loading displacement.txt:")
				print("displacement.txt file is missing or invalid.")
				sys.exit()

	def setCoordinatesOutput(self, inverseX = True, inverseY = False, inverseZ = False):
		self.inverseX = inverseX
		self.inverseY = inverseY
		self.inverseZ = inverseZ

		if(self.applyDisplacement):
			if(inverseX):
				self.displacementArray[0] = -self.displacementArray[0]
			if(inverseY):
				self.displacementArray[1] = -self.displacementArray[1]
			if(inverseZ):
				self.displacementArray[2] = -self.displacementArray[2]


	def setAnglesOutput(self, inverseXAngle = False, inverseYAngle = True, inverseZAngle = True):
		self.inverseXAngle = inverseXAngle
		self.inverseYAngle = inverseYAngle
		self.inverseZAngle = inverseZAngle

	def setMarkersInput(self, inverseXMarker = False, inverseYMarker = True, inverseZMarker = True):
		self.inverseXMarker = inverseXMarker
		if(inverseXMarker):
			self.markerPoints[:,0] = -self.markerPoints[:,0]

		self.inverseYMarker = inverseYMarker
		if(inverseYMarker):
			self.markerPoints[:,1] = -self.markerPoints[:,1]

		self.inverseZMarker = inverseZMarker
		if(inverseZMarker):
			self.markerPoints[:,2] = -self.markerPoints[:,2]

	def setSnapshotConfig(self, maxSuccesses = 10, maxFailures = 100, maxMarkersBypass = False):
		self.maxSuccesses = maxSuccesses
		self.maxFailures = maxFailures
		self.maxMarkersBypass = maxMarkersBypass
		msg = ""
		try:
			msg = "Inputs MaxSuccesses and MaxFailures must be positive integers."
			assert(maxSuccesses > 0 and maxFailures > 0)
			msg = "camera_points.txt file is missing or invalid."
			self.realCoordinatesMatrix = np.loadtxt("camera_points.txt")
			msg = "camera_points.txt file has invalid amount of columns (must be exactly six)"
			assert len(self.realCoordinatesMatrix[0]) == 6

		except:
			print("Unexpected ", sys.exc_info()[0], "occurred while setting snapshot mode configuration:")
			print(msg)
			sys.exit()


	def processFrame(self):
		# grab the frame from the threaded video stream
		while(True):
			frame, repeated = self.vs.read()
			if(not repeated):
				break

		# detect aruco markers in the frame
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		corners, idsM, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters, cameraMatrix=self.refinedMatrix, distCoeff=self.distCoeffs)

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

			success, rotation_vector, translation_vector = cv2.solvePnP(objPoints, imgPoints, self.refinedMatrix, self.distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
			translation_vector.mean()
			if(success):
				#solvePnP's x-axis rotation angle is of opposite sign relative to the Y coordinate

				if(self.applyDisplacement):
					for i in range(3):
						translation_vector[i] = translation_vector[i] + self.displacementArray[i]

				world_coordinates = self.__camera_to_world_coords(rotation_vector, translation_vector)
				rotation_vector[0] = -rotation_vector[0]

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

	def video(self, movingAverageFlag = False, movingAverageWindow = 20):

		if(movingAverageFlag):

			try:
				assert(movingAverageWindow > 0)
			except:
				print("Unexpected ", sys.exc_info()[0], "occurred while setting video mode configuration:")
				print("Moving average window must be a positive integer.")
				sys.exit()
			#initialize storage with zeros
			lastFramesArray = [[0 for x in range(6)] for y in range(movingAverageWindow)]
			#set weights depending on type
			if(movingAverageFlag == PoseDetector.SIMPLE_MOVING_AVERAGE):
				movingAverageWeights = [1 for x in range(movingAverageWindow)]
			if(movingAverageFlag == PoseDetector.WEIGHTED_MOVING_AVERAGE):
				movingAverageWeights = [(movingAverageWindow - x) for x in range(movingAverageWindow)]
			if(movingAverageFlag == PoseDetector.EXPONENTIAL_MOVING_AVERAGE):
				alpha = 2/(movingAverageWindow + 1)
				movingAverageWeights = [(1 - alpha)**x for x in range(movingAverageWindow)]

		#loop over captured frames until user interruption
		while (True):
			success, world_coordinates, rotation_vector, translation_vector, detectedMarkers, gray = self.processFrame()
			#show data in output image
			if(success):

				if(movingAverageFlag and movingAverageFlag != PoseDetector.MOVING_MEDIAN):
					lastFramesArray.insert(0, world_coordinates + rotation_vector)
					lastFramesArray.pop()
					movingAverageResult = np.ma.average(lastFramesArray, axis=0, weights=movingAverageWeights)
					world_coordinates = movingAverageResult[0:3]
					rotation_vector = movingAverageResult[3:6]

				if(movingAverageFlag == PoseDetector.MOVING_MEDIAN):
					lastFramesArray.insert(0, world_coordinates + rotation_vector)
					lastFramesArray.pop()
					movingAverageResult = np.median(lastFramesArray, axis=0)
					world_coordinates = movingAverageResult[0:3]
					rotation_vector = movingAverageResult[3:6]


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

		rawErrorsArray = np.empty((0,6), dtype=float)
		rawResultsArray = np.empty((0,6), dtype=float)
		nPoints = len(self.realCoordinatesMatrix)
		finalDetectedMarkers = []
		poseDetectedMarkersMatrix = []
		successPoses = np.empty(nPoints, dtype=bool)
		for i in range(nPoints):
			successes = 0
			errors_i = []
			print("\nNext pose:")
			print("X: " + str(self.realCoordinatesMatrix[i][0]))
			print("Y: " + str(self.realCoordinatesMatrix[i][1]))
			print("Z: " + str(self.realCoordinatesMatrix[i][2]))
			print("X-axis angle: " + str(self.realCoordinatesMatrix[i][3]))
			print("Y-axis angle: " + str(self.realCoordinatesMatrix[i][4]))
			print("Z-axis angle: " + str(self.realCoordinatesMatrix[i][5]))
			print("")
			input("Position camera and press Enter to continue.")
			print("Taking samples...\n")
			#take multiple shots of the point in question. if marker can't be detected, notify user
			failures = 0
			maxMarkers = 0
			maxMarkersBypass = self.maxMarkersBypass
			skipped = False
			tempRawResults = np.empty((0,6), dtype=float)
			tempRawErrors = np.empty((0,6), dtype=float)
			poseDetectedMarkers = []
			while(successes < self.maxSuccesses and not skipped):
				try:
					success, world_coordinates, rotation_vector, translation_vector, detectedMarkers, frame = self.processFrame()
				except:
					print("Unexpected ", sys.exc_info()[0],  " occured, aborting process.")
					if(i > 0):
						print("Current data will be stored in appropiate save files.\n")
						rawResultsArray = np.around(rawResultsArray, 3)
						self.__printResults(rawResultsArray, rawErrorsArray, successPoses, poseDetectedMarkersMatrix, finalDetectedMarkers, i)
						self.__printResultsRaw(rawResultsArray, rawErrorsArray, successPoses, i)
					self.stop()
					sys.exit()
				if(success):
					#if a new, previously undetected marker appeared, discard previous samples and start over, unless bypass activated
					newDetected, poseDetectedMarkers = self.__markersInArray(poseDetectedMarkers, detectedMarkers)
					if(newDetected):
						maxMarkers = len(poseDetectedMarkers)
						if(not maxMarkersBypass):
							tempRawResults = np.empty((0,6), dtype=float)
							tempErrorResults = np.empty((0,6), dtype=float)
							successes = 0
					#if bypass is activated, or it is not activated and all markers have been detected, store results
					if(maxMarkersBypass or len(detectedMarkers) == maxMarkers):
						failures = 0
						calculatedCoordinates = world_coordinates + rotation_vector
						successes += 1
						errorArray = self.realCoordinatesMatrix[i] - calculatedCoordinates
						for k in range(6):
							errorArray[k] = round(errorArray[k], 3)
						tempRawResults = np.vstack((tempRawResults, calculatedCoordinates))
						tempRawErrors = np.vstack((tempRawErrors, errorArray))

						if(not maxMarkersBypass and successes == self.maxSuccesses):
							print(str(maxMarkers) + " markers have been detected. Save results?[y/n]")
							while(True):
								inp = input("")
								if(inp == 'n'):
									print("Restarting pose configuration...")
									tempRawResults = np.empty((0,6), dtype=float)
									tempErrorResults = np.empty((0,6), dtype=float)
									successes = 0
									maxMarkers = 0
									poseDetectedMarkers = []
									input("Reposition camera and press Enter to continue.")
									print("Taking samples...")
									print("")
									break
								else:
									if(inp == 'y'):
										print("Saving pose results...")
										break
					else:
						failures = failures + 1
				else:
					failures = failures + 1
				if(failures >= self.maxFailures):
					if(maxMarkers == 0):
						print("Could not detect a marker from this pose. Would you like to try again? [y/n]:")
						while(True):
							inp = input("")
							if(inp == 'n'):
								print("Skipping current pose. Result will be filled with '-' for this pose.")
								skipped = True
								break
							else:
								if(inp == 'y'):
									failures = 0
									input("Reposition camera and press Enter to continue.")
									print("Taking samples...")
									print("")
									break
					else:
						failures = 0
						maxMarkers = 0
						poseDetectedMarkers = []
						print("Some markers could not be detected with enough consistency. Would you like to bypass maximum marker requirement? [y/n]:")
						while(True):
							inp = input("")
							if(inp == 'n'):
								input("Reposition camera and press Enter to continue.")
								print("Taking samples...")
								print("")
								break
							else:
								if(inp == 'y'):
									maxMarkersBypass = True
									print("Understood. Some samples likely won't detect the maximum amount of markers for this pose.")
									print("")
									break


			if(not skipped):
				#Notify that pose was successfully calculated
				successPoses[i] = True
				#Store results and errors of calculation
				rawErrorsArray = np.vstack((rawErrorsArray, tempRawErrors))
				rawResultsArray = np.vstack((rawResultsArray, tempRawResults))
				#Update list of detected markers, for all poses individually and collectively
				poseDetectedMarkers.sort()
				trash, finalDetectedMarkers = self.__markersInArray(finalDetectedMarkers, poseDetectedMarkers)
				poseDetectedMarkersMatrix.append(poseDetectedMarkers)
			else:
				failedArray = [0,0,0,0,0,0]
				successPoses[i] = False
				for k in range(self.maxSuccesses):
					rawErrorsArray = np.vstack((rawErrorsArray, failedArray))
					rawResultsArray = np.vstack((rawResultsArray, failedArray))
				poseDetectedMarkersMatrix.append([-1])
			print("Saved successfully.")

		rawResultsArray = np.around(rawResultsArray, 3)
		self.__printResults(rawResultsArray, rawErrorsArray, successPoses, poseDetectedMarkersMatrix, finalDetectedMarker)
		self.__printResultsRaw(rawResultsArray, rawErrorsArray, successPoses)

	def __printResults(self, rawResultsArray, rawErrorsArray, successPoses, poseDetectedMarkersMatrix, finalDetectedMarkers, midHalt = 0):
		file = open("results.txt", "a")

		if(midHalt > 0):
			nPoints = midHalt
			print("\nWARNING: Following data is incomplete, only the first ", midHalt, " poses are saved.\n")
		else:
			nPoints = len(self.realCoordinatesMatrix)
		#save outputs with context and filters

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

		posesList = self.realCoordinatesMatrix.tolist()

		for i in range(len(posesList)):
			strPoint = [str(point) for point in posesList[i]]
			joinPoints = " ".join(strPoint)
			file.write(str(i+1) + ": ")
			file.writelines(joinPoints)
			file.write("\n")

		file.write("\nDetected markers in each pose:\n")

		for i in range(nPoints):
			file.write(str(i+1) + ": ")
			if successPoses[i]:
				strPoint = [str(int(point)) for point in poseDetectedMarkersMatrix[i]]
				joinPoints = ", ".join(strPoint)
				file.writelines(joinPoints)
				file.write("\n")
			else:
				file.write("-\n")

		file.write("\nEstimation errors:\n")

		file.write("\nRaw results:\n")

		for i in range(nPoints):
			if(successPoses[i]):
				for j in range(self.maxSuccesses):
					file.write(str(i+1) + "." + str(j+1) + ": ")
					strPoint = [str(point) for point in rawResultsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.writelines(joinPoints)
					file.write("\n")
			else:
				for j in range(self.maxSuccesses):
					file.write(str(i+1) + "." + str(j+1) + ": ")
					file.write("- - - - - -\n")

		file.write("\nRaw errors:\n")

		for i in range(nPoints):
			if(successPoses[i]):
				for j in range(self.maxSuccesses):
					file.write(str(i+1) + "." + str(j+1) + ": ")
					strPoint = [str(point) for point in rawErrorsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.writelines(joinPoints)
					file.write("\n")
			else:
				for j in range(self.maxSuccesses):
					file.write(str(i+1) + "." + str(j+1) + ": ")
					file.writelines("- - - - - -\n")

		file.close()
		print("")
		print("Pose estimation results successfully saved in results.txt")

	def __printResultsRaw(self, rawResultsArray, rawErrorsArray, successPoses, midHalt = 0):
		file = open("raw_results.txt", "a")
		if(midHalt > 0):
			nPoints = midHalt
		else:
			nPoints = len(self.realCoordinatesMatrix)

		for i in range(nPoints):
			if(successPoses[i]):
				for j in range(self.maxSuccesses):
					strPoint = [str(point) for point in rawResultsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.writelines(joinPoints)
					file.write("\n")
			else:
				for j in range(self.maxSuccesses):
					file.write("- - - - - -\n")
		file.close()

		file = open("raw_errors.txt", "a")

		for i in range(nPoints):
			if(successPoses[i]):
				for j in range(self.maxSuccesses):
					strPoint = [str(point) for point in rawErrorsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.writelines(joinPoints)
					file.write("\n")
			else:
				for j in range(self.maxSuccesses):
					file.write("- - - - - -\n")

		file.close()
		print("")
		print("Raw data successfully saved in raw_results.txt and raw_errors.txt.")

	def stop(self):
		self.vs.stop()

	def __checkMarkerPoints(self, mtx):
		#Check correct shape of matrix
		if(mtx.shape[0]%4 != 0):
			return False, "Number of rows is not a multiple of 4: some markers have missing or excess corners."
		if(mtx.shape[1] != 4):
			return False, "Number of columns is different than 4. Notation is X Y Z ID."

		#Check correct structure of matrix
		checkedMarkers = []
		for i in range(0,mtx.shape[0],4):
			if(mtx[i][3] == mtx[i+1][3] == mtx[i+2][3] == mtx[i+3][3]):
				if(mtx[i][3] not in checkedMarkers):
					checkedMarkers.append(mtx[i][3])
				else:
					return False, ("Marker of ID " + str(int(mtx[i][3])) + " at row " + str(i+1) + " is duplicated.")
			else:
				return False, ("Marker at rows " + str(i+1) + "-" + str(i+4) + " has conflicting IDs.")

		return True, ""


	def __markersInArray(self, markersArray, newMarkers):
		newOnes = False
		for marker in newMarkers:
			if marker not in markersArray:
				markersArray.append(marker)
				newOnes = True
		return newOnes, markersArray

	def __camera_to_world_coords(self, rotation_vector, translation_vector):

		rotation_matrix, trash = cv2.Rodrigues(rotation_vector)

		world_coordinates_numpy = np.dot(-(np.linalg.inv(rotation_matrix)),translation_vector)

		#pass from numpy to list and undo translation's inverse output
		world_coordinates = [0,0,0]
		world_coordinates[0] = -world_coordinates_numpy[0][0]
		world_coordinates[1] = -world_coordinates_numpy[1][0]
		world_coordinates[2] = -world_coordinates_numpy[2][0]

		return world_coordinates


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--Port", type=int, default=0,
	help="Camera port to use (for more info, run testports.py), port 0 by default")

ap.add_argument('-rb', '--RecordBox', nargs='+', type=int, default=-1,
	help="Capture a box of the computer screen for video input instead of a camera. Coordinates are top, left, bottom and right edges respectively (top-left corner being (0,0)). This option overrides port.")

ap.add_argument("-dc", "--DefaultCalibration", default=False, action='store_true',
	help="Sets the camera's intrinsic parameters to defaults, skipping calibration. False by default.")

ap.add_argument("-ad", "--ApplyDisplacement", default=False, action='store_true',
	help="Applies displacement indicated by displacement.txt to results.")

ap.add_argument("-o", "--ShowOriginals", default=False, action='store_true',
	help="Shows coordinates directly calculated by OpenCV, relative to camera's orientation (only applicable in video mode)")

ap.add_argument("-ma", "--MovingAverage", choices=['None', 'Simple', 'Weighted', 'Exponential', 'Median'], default='None',
	help="Apply a moving average to smooth results (only applicable in video mode)")

ap.add_argument("-maw", "--MovingAverageWindow", type=int, default=20,
	help="Window size of moving average if any selected, 20 by default (only applicable in video mode)")

ap.add_argument("-s", "--Snapshot", default=False, action='store_true',
	help="Turns on snapshot mode (requires camera_points.txt, outputs results and errors in results.txt)")

ap.add_argument("-xs", "--MaxSuccesses", type=int, default=10,
	help="Number of required successful frames for each shot in snapshot mode, 10 by default (only applicable in snapshot mode)")

ap.add_argument("-xf", "--MaxFailures", type=int, default=100,
	help="Number of required consecutive failed frames to trigger warning in shapshot mode, 100 by default (only applicable in snapshot mode)")

ap.add_argument("-xmb", "--MaxMarkersBypass", default=False, action='store_true',
	help="Snapshot mode won't require every sample from every pose to have the maximum amount of markers detected, false by default (only applicable in snapshot mode)")

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
	help="Inverts X coordinate of marker inputs. (Default: positive X-right, Y-up, Z-backwards)")

ap.add_argument("-iYM", "--InverseYMarker", default=True, action='store_false',
	help="Inverts X coordinate of marker inputs. (Default: positive X-right, Y-up, Z-backwards)")

ap.add_argument("-iZM", "--InverseZMarker", default=True, action='store_false',
	help="Inverts X coordinate of marker inputs. (Default: positive X-right, Y-up, Z-backwards)")

args = vars(ap.parse_args())

pd = PoseDetector(port=args['Port'], boundingBox=args['RecordBox'], showOriginals=args['ShowOriginals'], defaultCalibration = args['DefaultCalibration'], applyDisplacement = args['ApplyDisplacement'])
pd.setCoordinatesOutput(inverseX = args['InverseX'], inverseY = args['InverseY'], inverseZ = args['InverseZ'])
pd.setAnglesOutput(inverseXAngle = args['InverseXAngle'], inverseYAngle = args['InverseYAngle'], inverseZAngle = args['InverseZAngle'])
pd.setMarkersInput(inverseXMarker = args['InverseXMarker'], inverseYMarker = args['InverseYMarker'], inverseZMarker = args['InverseZMarker'])
if(not args['Snapshot']):
	flag = False
	if(args['MovingAverage'] == 'Simple'):
		flag = PoseDetector.SIMPLE_MOVING_AVERAGE
	if(args['MovingAverage'] == 'Weighted'):
		flag = PoseDetector.WEIGHTED_MOVING_AVERAGE
	if(args['MovingAverage'] == 'Exponential'):
		flag = PoseDetector.EXPONENTIAL_MOVING_AVERAGE
	if(args['MovingAverage'] == 'Median'):
		flag = PoseDetector.MOVING_MEDIAN
	pd.video(movingAverageFlag = flag, movingAverageWindow = args['MovingAverageWindow'])
else:
	pd.setSnapshotConfig(maxSuccesses=args['MaxSuccesses'], maxFailures = args['MaxFailures'], maxMarkersBypass=args['MaxMarkersBypass'])
	pd.snapshots()

print("Closing software...")
pd.stop()
cv2.destroyAllWindows()
