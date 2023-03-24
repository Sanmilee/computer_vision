import numpy as np
import cv2
import sys

imagePath = "little_mix.jpg"
cascPath = "haarcascade_frontalface_default.xml"

# cascade for detecting faces provided by OpenCV
faceCascade = cv2.CascadeClassifier(cascPath)


# read the image and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect faces in the image --- scaleFactor=1.2
# detectMultiScale function is a general function to detects objects - we call in on face cascade
# scaleFactor - compensates for the proximity of the image to the camera
# minNeighbors defines how many objects are detected near the current one
# minSize gives the size of each window.
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)


print("Found {0} faces!".format(len(faces)))


# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Faces found", image)
cv2.waitKey(0)
