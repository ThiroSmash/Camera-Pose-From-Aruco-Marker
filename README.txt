	DEPENDENCIES

Python 3.9.1+
numpy 1.19.3+
imutils 0.5.3+
OpenCV with extended modules for python 4.4+
	-A pre-compiled but unsupported version of OpenCV can be quickly installed with pip, it is compatible with this program.

	USE INSTRUCTIONS

	SETUP
First, you need "chessboard_pattern.png" and any marker created by MarkerCreator.py printed, preferably in a blank white paper.
Make sure that the marker is 240x240 pixels = 6.35cm square, otherwise the calculations will be all wrong!

	CAMERA CALIBRATION
Plug into your device the camera that you wish to use. Then, run CameraCalibration.py, indicating the port of the camera with the --port argument (if you don't know which one it is, run testports.py so see which ones are available).

Show your printed chessboard to the camera. When a valid sample image is detected, it will show the corners in the image output, as well as notify through the default stream output. After a grace period of three seconds, it will start scanning for another sample.

By default, this process will be repeated ten times. The amount of times can be changed with the --iterations argument. Ten is the minimum recommended by OpenCV, however I would recommend around thirty. Make sure that either the camera or the chessboard pattern have different positions and orientations for a better calibration process. Also keep the chessboard as flat as possible (use something like a hardcover book as support if you intend to leave the camera stationary).

After all the samples have been collected, the program will show the distortion parameters as well as the camera matrix through the default output stream, as well as create txt files containing them locally.

	MAIN PROGRAM
To start the program, run ArucoMarkerDistance.py, once again indicating the port by --port. The camera calibration must have been done beforehand, otherwise the program won't find the required txt files with the camera's intrinsic parameters.

If succesful, the program will output the images and coordinates plus rotation angles of the camera if it is detecting a marker.