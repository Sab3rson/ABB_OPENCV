from ultralytics import YOLO
import cv2

#Get the path of best trained model
model = YOLO(r"G:\GithubRepos\ABB_OPENCV\runs\detect\LatestTrained\weights\best.pt")



#Capture frames from my webcam
vid_capture = cv2.VideoCapture(0)

while True:
    ret, frame = vid_capture.read() #ret checks if frame exists
    if not ret:
        break

    #apparently YOLO returns a batch which is an array
    #because this becomes useful when predicting multiple
    #images at once
    #since I am only inputting one image at a time
    #it will return a batch of size 1
    results = model(frame)[0]

    for box in results.boxes:
         #XY coordinates of box probably relative to camera image
         # origin is probably top left corner
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = box.conf[0].item()
        cls = int(box.cls[0].item()) # Class ID (Cylinder)


        #Draw the rectangle box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #red label if innacurate
        if confidence < 0.8:
        #Put label with confidence
            label = f"{model.names[cls]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0,0), 2)
        else:
            label = f"{model.names[cls]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (0, 255,0), 2)
    
    #show frame
    cv2.imshow("Webcam", frame)

    #Exit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release resources
vid_capture.release()
cv2.destroyAllWindows()