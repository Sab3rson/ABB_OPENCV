from ultralytics import YOLO
import multiprocessing

def run_training():
    model = YOLO("G:\GithubRepos\ABB_OPENCV\yolo11n.pt")  # fresh model

    model.train(
        data="G:/GithubRepos/ABB_OPENCV/objDetection/circleonlyconfig.yaml",
        epochs=100,
        batch=2,
        imgsz=640,
        device=0,
        workers=0,
        project="G:/GithubRepos/ABB_OPENCV/runs/detect",
        name="CircleOnlyTest"
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_training()