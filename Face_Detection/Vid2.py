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



video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faceCascade
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        if w > 250:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

    # smileCascade
    smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        cv2.putText(frame, 'Smile', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

    # eyeCascade
    eyes = eyeCascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.putText(frame, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)

    cv2.putText(frame, 'Number of Faces : ' + str(len(faces)),
                (40, 40), font, 1, (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()
