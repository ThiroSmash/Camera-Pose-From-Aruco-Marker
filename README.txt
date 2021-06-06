	DEPENDENCIES

Python 3.9.1+
numpy 1.19.3+
imutils 0.5.3+
OpenCV with extended modules for python 4.4+


	USE INSTRUCTIONS

	SETUP
First, you need "chessboard_pattern.png" and any marker created by MarkerCreator.py printed, preferably in a blank white paper.
	The size of the images made by MarkerCreator.py is of 120x120 pixels by default (can be changed with --size), which translates to 3.175x3.175 cm. The markers don't need to be this size, but it has to be taken into account for correct calculations.

Additionally, you must create a txt file called "marker_points.txt". This file will contain a matrix with the real world points of each of the markers that you wish to use. Each row has the format:
X Y Z ID
to represent one corner. The rows must be ordered representing the corners clock-wise, starting top-left (so top-left, top-right, botton-right, bottom-left). Also, the four corners of each marker must be consecutive (all corners of one marker, then all corners of another marker, then all corners of another marker, etc).
	NOTE: OpenCV treats X's right, Y's down and Z's forward directions as positive when reading coordinates. However in this program by default, it is X-right, Y-up and Z-backwards in order to align with the default output coordinate system. Any and all systems can be customised through parameters, see --help for more info.

	To-do: Python file to create markerPoints.txt from user inputs

You must know which port your desired camera is using. For that, testports.py will show all the currently available and working ports.

(OPTIONAL)
In order to use the snapshot mode (more info below), you must create a txt file called "camera_points.txt". Each row of this file represents a camera pose and must contain a translation vector followed by a rotation vector (X coord, Y coord, Z coord, X-axis angle, Y-axis angle, Z-axis angle). More information in 'Main Program' section.

	CAMERA CALIBRATION
Plug into your device the camera that you wish to use. Then, run CameraCalibration.py, indicating the port of the camera with the --port argument.

Show your printed chessboard to the camera. When a valid sample image is detected, it will show the corners in the image output, as well as notify through the default stream output. After a grace period of three seconds, it will start scanning for another sample.

By default, this process will be repeated ten times. The amount of times can be changed with the --iterations argument. Ten is the minimum recommended by OpenCV, however I would recommend around thirty. Make sure that either the camera or the chessboard pattern have different positions and orientations for a better calibration process. Also keep the chessboard as flat as possible (use something like a hardcover book as support if you intend to leave the camera stationary).

After all the samples have been collected, the program will show the distortion parameters as well as the camera matrix through the default output stream, as well as create txt files containing them locally.

	MAIN PROGRAM
To start the program, run PoseDetection.py, once again indicating the port by --port. The camera calibration must have been done beforehand, otherwise the program won't find the required txt files with the camera's intrinsic parameters (default parameters option still to-do).

The program has two different "modes": video mode and snapshot mode.

In video mode it will simply calculate the camera pose and show the results in-real-time through video. This is the mode by default.

In snapshot mode, the program will read the camera poses that the user has defined in "camera_points.txt". One by one, the program will ask the user to position the camera in one of the poses they defined in the file, and press Enter.
Once Enter is pressed, the program will take a few image samples (ten by default, can be customised with --maxSuccesses or -x), calculate the positions and store the results for later output. If --MakeStats option was selected, it will calculate a mean and a median value for each pose; otherwise only the raw results will be written.
The results will be printed into a file "results.txt", with all the information regarding defined camera poses, marker poses and the pose estimations.
The purpose of snapshot mode is to easily and quickly produce dozens of pose estimations, for a better reflection of the program's precision.
