import cv2

net = cv2.dnn.readNet()

vidCapture = cv2.VideoCapture(0)

while True:
# frame is each frame of webcame and ret
# checks whether frame exists or not
    ret, frame = vidCapture.read() 

    cv2.imshow("frame", frame)
    cv2.waitKey(1)