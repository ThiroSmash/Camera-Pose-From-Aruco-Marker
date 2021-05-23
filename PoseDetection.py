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

	def __init__(self, port=0, showOriginals=False):

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

		self.markerIds = mtx[::4,3]

		self.showOriginals = showOriginals

		#self.setCoordinatesOutput()
		#self.setAnglesOutput()
		#self.setMarkersInput()

	def setCoordinatesOutput(self, inverseX = True, inverseY = False, inverseZ = False):
			self.inverseX = inverseX
			self.inverseY = inverseY
			self.inverseZ = inverseZ

	def setAnglesOutput(self, inverseXAngle = False, inverseYAngle = True, inverseZAngle = True):
			self.inverseXAngle = inverseXAngle
			self.inverseYAngle = inverseYAngle
			self.inverseZAngle = inverseZAngle

	def setMarkersInput(self, inverseXMarker=False, inverseYMarker=True, inverseZMarker=True):
		self.inverseXMarker = inverseXMarker
		if(inverseXMarker):
			self.markerPoints[:,0] = -self.markerPoints[:,0]

		self.inverseYMarker = inverseYMarker
		if(inverseYMarker):
			self.markerPoints[:,1] = -self.markerPoints[:,1]

		self.inverseZMarker = inverseZMarker
		if(inverseZMarker):
			self.markerPoints[:,2] = -self.markerPoints[:,2]

	def setSnapshotConfig(self, rawOutputs=False, maxSuccesses=10, makeStats=False):
		self.rawOutputs = rawOutputs
		self.maxSuccesses = maxSuccesses
		self.makeStats = makeStats


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
				world_coordinates = self.__camera_to_world_coords(rotation_vector, translation_vector)

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
		rawErrorsArray = np.empty((0,6), dtype=float)
		rawResultsArray = np.empty((0,6), dtype=float)
		nPoints = len(realCoordinatesMatrix)
		finalDetectedMarkers = []
		successPoses = np.empty(nPoints, dtype=bool)
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
					for k in range(6):
						error[k] = round(error[k], 3)
					rawResultsArray = np.vstack((rawResultsArray, calculatedCoordinates))
					rawErrorsArray = np.vstack((rawErrorsArray, error))
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


			if(not failed):
				successPoses[i] = True
			else:
				failedArray = ['-','-','-','-','-','-']
				successPoses[i] = False
				for k in range(self.maxSuccesses):
					rawErrorsArray = np.vstack((rawErrorsArray, failedArray))
					rawResultsArray = np.vstack((rawResultsArray, failedArray))

		self.__printResults(rawResultsArray, rawErrorsArray, realCoordinatesMatrix, nPoints, successPoses, finalDetectedMarkers)


		#save results in txt

	def __printResults(self, rawResultsArray, rawErrorsArray, realCoordinatesMatrix, nPoints, successPoses, finalDetectedMarkers):
		file = open("results.txt", "a")

		#save outputs with context and filters
		if(not self.rawOutputs):

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

			#if user asked for statistic values
			if(self.makeStats):

				meanErrorsArray = np.empty((0,6), dtype=float)
				medianErrorsArray = np.empty((0,6), dtype=float)
				stdErrorsArray = np.empty((0,6), dtype=float)
				#for each pose
				for i in range(nPoints):

					#calculate mean and standard deviation
					if(successPoses[i]):
						#extract results of pose
						results_i = rawResultsArray[i*self.maxSuccesses:(i+1)*self.maxSuccesses][:].astype(np.float)
						#calculate mean and round to 3 decimals
						meanResult_i = np.mean(results_i, axis=0)
						for k in range(6):
							meanResult_i[k] = round(meanResult_i[k], 3)
						meanError = realCoordinatesMatrix[i] - meanResult_i
						#add result to mean errors array
						meanErrorsArray = np.vstack((meanErrorsArray, meanError))

						#calculate median and round to 3 decimals
						medianResult_i = np.median(results_i, axis=0)
						for k in range(6):
							medianResult_i[k] = round(medianResult_i[k], 3)
						medianError = realCoordinatesMatrix[i] - medianResult_i
						#add result to median errors array
						medianErrorsArray = np.vstack((medianErrorsArray, medianError))
						#extract absolute errors of raw results
						errors_i_abs = np.absolute(rawErrorsArray[i*self.maxSuccesses:(i+1)*self.maxSuccesses][:].astype(np.float))
						#calculate standard deviation of raw errors, round to 3 decimals
						stdError = np.std(errors_i_abs, axis=0)
						for k in range(6):
							stdError[k] = round(stdError[k], 3)
						#add result to std errors array
						stdErrorsArray = np.vstack((stdErrorsArray, stdError))

					else:
						meanErrorsArray = np.vstack((meanErrorsArray, ['-','-','-','-','-','-']))
						medianErrorsArray = np.vstack((medianErrorsArray, ['-','-','-','-','-','-']))
						stdErrorsArray = np.vstack((stdErrorsArray, ['-','-','-','-','-','-']))

				file.write("\nWith mean filter:\n")
				for i in range(nPoints):

					strPoint = []

					if(successPoses[i]):
						floatPoints = [float(point) for point in meanErrorsArray[i]]
						print(floatPoints)
						for k in range(len(floatPoints)):
							floatPoints[k] = round(floatPoints[k], 3)
						print(floatPoints)
						strPoint = [str(point) for point in floatPoints]
					else:
						strPoint = [str(point) for point in meanErrorsArray[i]]

					joinPoints = " ".join(strPoint)

					file.write(str(i+1) + ": ")
					file.writelines(joinPoints)
					file.write("\n")

				file.write("\nWith median filter:\n")

				for i in range(nPoints):

					strPoint = []

					if(successPoses[i]):
						floatPoints = [float(point) for point in medianErrorsArray[i]]
						for k in range(len(floatPoints)):
							floatPoints[k] = round(floatPoints[k], 3)
						strPoint = [str(point) for point in floatPoints]
					else:
						strPoint = [str(point) for point in meanErrorsArray[i]]

					joinPoints = " ".join(strPoint)

					file.write(str(i+1) + ": ")
					file.writelines(joinPoints)
					file.write("\n")

				file.write("\nStandard deviation in each pose:\n")

				for i in range(nPoints):

					strPoint = []

					if(successPoses[i]):
						floatPoints = [float(point) for point in stdErrorsArray[i]]
						for k in range(len(floatPoints)):
							floatPoints[k] = round(floatPoints[k], 3)
						strPoint = [str(point) for point in floatPoints]
					else:
						strPoint = [str(point) for point in meanErrorsArray[i]]

					joinPoints = " ".join(strPoint)

					file.write(str(i+1) + ": ")
					file.writelines(joinPoints)
					file.write("\n")

			#for i in range(self.maxSuccesses*nPoints):
			#	for j in range(6):

			rawResultsArray = np.around(rawResultsArray, 3)

			file.write("\nRaw results:\n")

			for i in range(nPoints):
				for j in range(self.maxSuccesses):

					strPoint = [str(point) for point in rawResultsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					file.write(str(i+1) + "." + str(j+1) + ": ")
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

			file.write("\nRaw results:\n")

			for i in range(nPoints):
				for j in range(self.maxSuccesses):

					strPoint = [str(point) for point in rawResultsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					#file.write(str(i+1) + "." + str(j+1) + ": ")
					file.writelines(joinPoints)
					file.write("\n")

			file.write("\nRaw errors:\n")

			for i in range(nPoints):
				for j in range(self.maxSuccesses):

					strPoint = [str(point) for point in rawErrorsArray[i*self.maxSuccesses + j]]
					joinPoints = " ".join(strPoint)
					#file.write(str(i+1) + "." + str(j+1) + ": ")
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

	def __camera_to_world_coords(self, rotation_vector, translation_vector):
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

ap.add_argument("-o", "--ShowOriginals", default=False, action='store_true',
		help="Shows coordinates directly calculated by OpenCV (relative to camera's orientation) (only applicable in video mode)")

ap.add_argument("-s", "--Snapshot", default=False, action='store_true',
	help="Turns on snapshot mode (requires camera_points.txt, outputs estimation errors in results.txt)")

ap.add_argument("-r", "--RawOutputs", default=False, action='store_true',
	help="Snapshot mode outputs raw error arrays without context data or filters (only applicable in snapshot mode)")

ap.add_argument("-st", "--MakeStats", default=False, action='store_true',
	help="Snapshot mode adds median, mean and standard deviation to results (overriden by RawOutputs) (only applicable in snapshot mode)")

args = vars(ap.parse_args())

pd = PoseDetector(port=args['Port'], showOriginals=args['ShowOriginals'])
pd.setCoordinatesOutput(inverseX = args['InverseX'], inverseY = args['InverseY'], inverseZ = args['InverseZ'])
pd.setAnglesOutput(inverseXAngle = args['InverseXAngle'], inverseYAngle = args['InverseYAngle'], inverseZAngle = args['InverseZAngle'])
pd.setMarkersInput(inverseXMarker = args['InverseXMarker'], inverseYMarker = args['InverseYMarker'], inverseZMarker = args['InverseZMarker'])
if(not args['Snapshot']):
	pd.video()
else:
	pd.setSnapshotConfig(rawOutputs = args['RawOutputs'], maxSuccesses=args['MaxSuccesses'], makeStats=args['MakeStats'])
	pd.snapshots()

print("Closing software...")
pd.stop()
cv2.destroyAllWindows()
