# yolov5を使って物体検出を行うスクリプト。
# yolov5, weightを変更する場合は、 yolov5_path, weight_pathを変更してください。

# Usage:
# python detect_by_yolov5.py [検出対象の画像フォルダパス] [出力フォルダ名]
# python detect_by_yolov5.py ../res/data/img/ ../res/out/detect




# Standard Libraries



import sys
import os
import pathlib
import json
import cv2
# External Libraries
import torch
import pandas as pd
import torchvision
from torchvision.utils import save_image


class YoloV5Detection():

    def __init__(self):
        pwd_path = pathlib.Path.cwd()

        yolov5_path = str(pwd_path) + '/../yolov5'
        weight_path = str(pwd_path) + '/weight/1208-best.pt';


        self.model = torch.hub.load(yolov5_path, 'custom', source='local', weithgt_path, force_reload=True)

    def detect(self, filepath, outname):
        # Test Image

        # Inference
        results = self.model(filepath)

        # results.show()
        pwd_path = str(pathlib.Path.cwd())

        # mAP計算で使えるかも
        # torch.hub

        # results.render()
        # cv2.imwrite(pwd_path + '/results/20230403/point1', results.ims[0])

        data = results.pandas().xyxy[0].to_json(orient="records")
        return data

    def save_results(self, results, outname):
        pwd_path = pathlib.Path.cwd()
        save_location = str(pwd_path) + outname + '/'
        if not os.path.exists(save_location):
            os.makedirs(save_location)

        file_name = 'bbox_detections.json'
        save_location = os.path.join(save_location, file_name)

        data = {
            'yolo': results
        }

        with open(save_location, "w+") as output_file:
            json.dump(data, output_file, indent=4, sort_keys=True, separators=(',', ': '))

def parse_args(argv):
    target_folder_path = ''
    if len(sys.argv) < 2:
        raise ValueError ("Missing arguments")
    else:
        target_folder_path = sys.argv[1]

    return target_folder_path

def format_results(results):
    results = json.loads(results)
    formatted_bbox_info = []

    for bbox in results:
        formatted_bbox_info.append({
            'xmin': int(bbox['xmin']),
            'ymin': int(bbox['ymin']),
            'xmax': int(bbox['xmax']),
            'ymax': int(bbox['ymax']),
            'confidence': int(bbox['confidence']),
            'class': int(bbox['class']),
            'name': bbox['name']
        })

    return formatted_bbox_info






if __name__ == '__main__':
    # Model
    net = YoloV5Detection()

    # 検出フォルダのpath読込み
    input_folder_path = parse_args(sys.argv)
    output = sys.argv[2]

    bbox_info_list = []

    for filename in os.listdir(input_folder_path):
        filepath = os.path.join(input_folder_path, filename)

        # if os.path.isfile(filepath) and filename.endswith(".jpg") or filename.endswith(".png"):
        if os.path.isfile(filepath) and filename.endswith(".jpg"):
            print(filepath)
            bbox_info_detect = format_results(net.detect(filepath, output))

            bbox_info_list.append({
                'file_name': filename,
                'file_path': filepath,
                'bbox_info': bbox_info_detect
            })

    # Save results
    net.save_results(bbox_info_list, output)
