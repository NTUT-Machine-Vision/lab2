import requests
import os
from typing import List
import argparse


class NTUT_ServiceClient:
    def __init__(self, host, port):
        """
        初始化 服務客戶端

        Args:
            host: 伺服器的 IP 位址或網域名稱
            port: 伺服器的連接埠
        """
        self.base_url = f"http://{host}:{port}"

    def upload_img(self, file_path: str, model: str) -> dict:
        """上傳 img 檔案到伺服器"""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(f"{self.base_url}/upload-img/{model}", files=files)
        return response.json()

    def get_image_list(self, pdf_name: str) -> List[str]:
        """取得切割後的圖片檔案清單"""
        response = requests.get(f"{self.base_url}/get-images/{pdf_name}")
        result = response.json()
        return result.get("images", [])

    def upload_model(self, file_path: str) -> str:
        """上傳 img 檔案到伺服器"""
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(f"{self.base_url}/upload-model/", files=files)
        return response.json()

    def download_image(self, file_name: str, save_path: str, base: str) -> bool:
        """下載指定的圖片檔案"""
        print(f"Calling api {self.base_url}/download-image/{file_name}/{base}")
        response = requests.get(
            f"{self.base_url}/download-image/{file_name}/{base}", stream=True
        )
        dirfolder, basename = os.path.split(file_name)
        dirfolder = os.path.join(save_path, base)
        if os.path.exists(dirfolder) == False:
            os.makedirs(dirfolder)
        save_path = os.path.join(dirfolder, basename)
        if response.status_code == 200:
            try:
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                return False


# 使用範例
def run_service(host: str, port: int, model: str, file: str, output_path: str):
    client = NTUT_ServiceClient(host=host, port=port)

    # 上傳 PDF
    result = client.upload_model(model)
    print("上傳模型結果:", result)

    result = client.upload_img(file, result)
    print("上傳測試影像結果:", result)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    baseF = result["baseF"]
    output_path = os.path.join(current_dir, output_path)
    for rr in result["processed_images"]:
        success = client.download_image(
            os.path.basename(rr), save_path=output_path, base=baseF
        )
        print(f"下載 {os.path.basename(rr)}: {'成功' if success else '失敗'}")
    files = [result["summary_file_path"], result["time_path"]]
    for f in files:
        success = client.download_image(os.path.basename(f), output_path, base=baseF)
        print(f"下載 {os.path.basename(f)}: {'成功' if success else '失敗'}")


if __name__ == "__main__":
    host = "140.124.181.195"
    port = 8080 # adjust to 8010 if using yolo
    '''
    # yolo
    models = [
        "yolo_models/yolo11n_1000epochs_adamw_best.pt",
        "yolo_models/yolo11n_1000epochs_original_best.pt",
        "yolo_models/yolo11n_1000epochs_batch100_best.pt",
        "yolo_models/yolo11n_1000epochs_adamw_last.pt",
        "yolo_models/yolo11n_1000epochs_original_last.pt",
        "yolo_models/yolo11n_1000epochs_batch100_last.pt",
    ]
    '''
    # rtdetr
    # '''
    models = [
        "rtdetr_models/rtdetr_1000epochs_original_best.pt",
        "rtdetr_models/rtdetr_1000epochs_original_last.pt",
    ]
    # '''
    files = [
        f"image/{i}.png" for i in range(1, 11)
    ]
    for model in models:
        for file in files:
            run_service(host, port, model, file, "output")

