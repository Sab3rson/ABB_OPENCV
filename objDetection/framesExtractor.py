import os
import cv2
from pathlib import Path

video_path = r'C:\Users\poreb\Downloads\IMG_0509.MOV'

video = cv2.VideoCapture(video_path)

out_dir = Path(r"G:\GithubRepos\ABB_OPENCV\objDetection\trainingIMG")
out_dir.mkdir(parents=True, exist_ok=True)


ms = 0
interval = 100
i = 131

ret = True
while ret:
    video.set(cv2.CAP_PROP_POS_MSEC, ms)
    ret, frame = video.read()

    if not ret:
        break

    filename = out_dir / f"{i}.jpg"
    cv2.imwrite(str(filename), frame)
    i+=1
    ms += interval

video.release()