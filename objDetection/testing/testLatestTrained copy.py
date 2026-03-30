from ultralytics import YOLO
import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()

#Get the path of best trained model
model = YOLO(r"/media/kporeb/689CE3DA9CE3A0B4/ABB_OPENCV/runs/detect/ChipsTraining1/Latest/weights/best.pt")

#Capture frames from my webcam

center_x = 0
center_y = 0

color = (0,0,0)

frame_count = 0

results = None

while True:
    frame = picam2.capture_array()
    
    cv2.imshow("Camera", frame)

    #apparently YOLO returns a batch which is an array
    #because this becomes useful when predicting multiple
    #images at once
    #since I am only inputting one image at a time
    #it will return a batch of size 1
    
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
    
    frame_count += 1
    
    if frame_count % 2 == 0:
        small_frame = cv2.resize(frame, (200,100))
        results = model(small_frame)[0]
        print(results)
        
    
    print(type(model(frame)))
    results = model(frame)[0]
    if results is not None:
        for box in results.boxes:
         #XY coordinates of box probably relative to camera image
         # origin is probably top left corner
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = box.conf[0].item()
            cls = int(box.cls[0].item()) # Class ID (Circles, Squares, etc)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

        #Draw the rectangle box

            if confidence >= 0.8:
                color = (0,255,0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (color), 2)

        #Center point
            cv2.circle(frame, (center_x, center_y), 5, (0,0,255), -1)

            label = f"{model.names[cls]} {confidence:.2f}"

            cv2.putText(frame, label, (x1, y1 -10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    color,
                    2)


    

    #Exit on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release resources
picam2.end()
cv2.destroyAllWindows()
