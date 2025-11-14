# main.py
from ultralytics import YOLO
import multiprocessing

def run_training():
    model = YOLO(r"G:\GithubRepos\ABB_OPENCV\runs\detect\train14\weights\last.pt")
    # example: set workers to a small number or 0 if you still see issues
    model.train(
        data=r"G:\GithubRepos\ABB_OPENCV\objDetection\config.yaml",
        epochs=500,
        batch=4,
        imgsz=540,
        device=0,       # use GPU 0
        workers=0,
               # try 4, lower to 0 if problems continue
    )

if __name__ == "__main__":
    # on Windows, include freeze_support to be safe when using spawn
    multiprocessing.freeze_support()

    # optional: explicitly set start method, but default 'spawn' on Windows is fine
    # multiprocessing.set_start_method('spawn', force=True)

    run_training()
