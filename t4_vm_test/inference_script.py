from ultralytics import RTDETR, YOLO
import pathlib
import os
import time
import tracemalloc


def inference(model_path, img_path, save_dir, model_type):

    tracemalloc.start()

    # Run inference and time the process
    start_time = time.time()
    if model_type == "YOLO":
        model = YOLO(model_path)
    elif model_type == "RTDETR":
        model = RTDETR(model_path)
    end_load_time = time.time()
    results = model.predict(img_path)
    end_predict_time = time.time()

    # Create save directory if it doesn't exist
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # Get the result (there will be only one result for a single image)
    assert len(results) == 1, "There should be only one result for a single image"
    result = results[0]

    # Save summary
    summary = result.summary()
    with open(pathlib.Path(save_dir) / "result.json", "w") as summary_file:
        summary_file.write(str(summary))

    # Save crop results
    result.save_crop(save_dir=save_dir, file_name="detection")

    # Save time information
    model_speed = (end_load_time - start_time) * 1000
    inference_speed = (end_predict_time - end_load_time) * 1000
    with open(pathlib.Path(save_dir) / "time.txt", "w") as f:
        f.write(
            f"model_path={model_path},\nimg_path={img_path},\nmodel_speed={model_speed} ms,\ninference_speed={inference_speed} ms\n"
        )

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Save memory information
    with open(pathlib.Path(save_dir) / "memory.txt", "w") as f:
        f.write(
            f"model_path={model_path},\nimg_path={img_path},\npeak = {peak / 10**6} MB\n"
        )

    return (
        pathlib.Path(save_dir) / "result.json",
        pathlib.Path(save_dir) / "time.txt",
    )


model_types = ["YOLO", "RTDETR"]
model_paths = {
    "YOLO": {
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_original/weights/best.pt": "original_best",
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_batch100/weights/best.pt": "batch100_best",
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_adamw/weights/best.pt": "adamw_best",
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_original/weights/last.pt": "original_last",
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_batch100/weights/last.pt": "batch100_last",
        "/home/ntut/CV_NTUT/Group2/runs/detect/yolo11n_1000epochs_adamw/weights/last.pt": "adamw_last",
    },
    "RTDETR": {
        "/home/ntut/CV_NTUT/Group2/rtdetr/runs/detect/rtdetr_1000epochs_original/weights/best.pt": "original_best",
        "/home/ntut/CV_NTUT/Group2/rtdetr/runs/detect/rtdetr_1000epochs_original/weights/last.pt": "original_last",
    },
}
img_paths = [
    f"/home/ntut/CV_NTUT/Group2/111590002/dataset/test/{i}.png" for i in range(1, 11)
]

if __name__ == "__main__":
    # Run inference for each combination of model and image
    for model in model_types:
        for model_path, model_name in model_paths[model].items():
            for img_path in img_paths:
                save_dir = f"/home/ntut/CV_NTUT/Group2/111590002/inference_results/{model}/{model_name}/img_{os.path.basename(img_path)}"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(f"Running inference for {model} on {img_path} with model {model_path}")
                inference(model_path, img_path, save_dir, model)
