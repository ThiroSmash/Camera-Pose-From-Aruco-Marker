import numpy as np
import cv2, time
import cv2.aruco as aruco
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--id", type=int, default=0,
	help="ID of the marker to be written")

args = vars(ap.parse_args())

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# second parameter is id number
# last parameter is total image size
img = aruco.drawMarker(aruco_dict, args["id"], 240)
name = "marker" + str(args["id"]) + ".jpg"

cv2.imwrite(name, img)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()