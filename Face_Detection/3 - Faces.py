
import cv2
import matplotlib.pyplot as plt
# import dlib
#from imutils import face_utils


cascPath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"
smilePath = "haarcascade_smile.xml"


faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)


# Detect a sample face on an image

# Load the image
#imagePath = "dude.png"
gray = cv2.imread("dude.png")

plt.figure(figsize=(12, 8))
plt.imshow(gray, cmap='gray')
plt.show()


# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    flags=cv2.CASCADE_SCALE_IMAGE
)


# For each face
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 3)


cv2.imshow("Faces found", gray)
cv2.waitKey(0)

