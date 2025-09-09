import cv2

#read webcam
webcam = cv2.VideoCapture(0)

# visualize webcam

while True:
    ret, frame = webcam.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
webcam.destroyAllWindows()