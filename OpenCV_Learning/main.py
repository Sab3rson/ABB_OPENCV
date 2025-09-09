import cv2

webcamCapture = cv2.VideoCapture(0)

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=100)

while True:
    ret, frame = webcamCapture.read()
    height, width, _ = frame.shape

    print(height, width)

    #extract region of interest
    roi = frame[300:1000, 200:1400]

    #object detection
    mask = object_detector.apply(roi)
    countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in countours:
        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 1000:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x + w, y + h), (0,255,0), 3)


    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Maks", mask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

webcamCapture.release()
cv2.destroyAllWindows()