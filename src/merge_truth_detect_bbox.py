# yolov5のdetection結果とtruthを比較するために、truthののbbxo情報を取得します。
# これbboxとtruthを比較したかったっぽい。




# Standard Libraries
import sys
import os
import json
import getopt
import argparse
import pathlib

# External Libraries
import torch
import cv2
import numpy as np

class MergeResults():
    def __init__(self):
        pwd_path = str(pathlib.Path.cwd())
        self.bbox_results_location = pwd_path + '/results/bbox_detections.json'
        self.truth_results_location = pwd_path + '/results/bbox-truth.json'
        self.merged_results_location = pwd_path + '/results/merge_truth_detect.json'
        self.visualized_results_location = pwd_path + '/results/truth'


    def merge_results(self, filename, filepath):
        input_bbox_file = open(self.bbox_results_location)
        input_truth_file = open(self.truth_results_location)

        bbox_results = json.load(input_bbox_file)['yolo']
        truth_results = json.load(input_truth_file)['yolact']

        result = {}
        result['file_path'] = filepath

        for data in bbox_results:
            if (data['file_name'] == filename):
                result['file_name'] = data['file_name']
                result['file_path'] = data['file_path']
                result['bbox_info'] = data['bbox_info']

        for data in mask_results:
            # result['mask_info'] = []
            if self.check_file_name(data['file_name'], filename):
                result['mask_info'] = data['mask_info']
             #   print(result['mask_info'])

        input_bbox_file.close()
        input_mask_file.close()

        return result

    def check_file_name(self, file1, file2):
        filename1 = os.path.splitext(file1)[0]
        filename2 = os.path.splitext(file2)[0]
        return filename1 == filename2

    def save_results(self, input_folder_path, merge_results):
        data = {
            'input_path': input_folder_path,
            'merged_data': merge_results
        }

        with open(self.merged_results_location, "w+") as output_file:
            json.dump(data, output_file)

    def visualize_results(self):
        input_merged_file = open(self.merged_results_location)
        merged_results = json.load(input_merged_file)['merged_data']

        # example of color profile: class_name
        color_profile = {
            "square_box_w_marker_front_side": (0, 0, 255),
            "square_box_front_side": (255,0,0),
            "square_box_back_side": (0,255,0),
            "small_rectangle_box_w_marker_front_side": (255,255,0),
            "small_rectangle_box_front_side": (0,255,255),
            "small_rectangle_box_back_side": (255,0,255),
            "big_rectangle_box_w_marker_front_side": (192,192,192),
            "big_rectangle_box_front_side": (128,128,128),
            "big_rectangle_box_back_side": (128,0,0),
            "connector": (128,0,128),
            "vertical_box_frontside_with_marker": (0, 0, 128),
            "vertical_box_backside": (0, 0, 64),
            "pipe": (0, 128, 128)
        }

        # print(merged_results)

        for result in merged_results:
            file_name = result['file_name']
            file_path = result['file_path']
            image_data = cv2.imread(file_path)
            overlay = image_data.copy()

            # Draw predicted bounding boxes
            bbox_results = json.loads(result['bbox_info'])
            for bbox in bbox_results:
                bbox_left = int(bbox['xmin'])
                bbox_top = int(bbox['ymin'])
                bbox_right = int(bbox['xmax'])
                bbox_bottom = int(bbox['ymax'])
                bbox_confidence = bbox['confidence']
                bbox_label = bbox['name']

                img_height, img_width, _ = image_data.shape
                thick = int((img_height + img_width) // 500)
                color = color_profile[bbox_label]

                cv2.rectangle(image_data, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, thick)
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org
                org = (bbox_left, bbox_top - 7)

                # fontScale
                fontScale = 1

                # Blue color in BGR
                # color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2
                image = cv2.putText(image_data, str(bbox_label), org, font,
                   fontScale, color, thickness, cv2.LINE_AA)
                # cv2.putText(image_data, str(bbox_label), (bbox_left, bbox_top - 7), 0, 1e-3 * img_height, color, 2)

            # Draw predicted mask results
            if 'mask_info' in result:
                mask_info = result['mask_info']
                for data in mask_info:
                    if not data['contour']:
                        continue
                    # mask_label = data['class_name']
                    mask_contour = np.array(data['contour'])

                    # mask_confidence = data['confidence']
                    cv2.fillPoly(overlay, [mask_contour], color=(255, 0, 0))
                # https://pystyle.info/opencv-find-contours/
                # https://qiita.com/chobaken/items/97e2bf9ad41129045ba0
                # https://campkougaku.com/2022/04/11/equirectangular1/#toc2

            alpha = 0.4

            new = cv2.addWeighted(overlay, alpha, image_data, 1 - alpha, 0)

            image_output_path = os.path.join(self.visualized_results_location, file_name)

            if not os.path.exists(self.visualized_results_location):
                os.makedirs(self.visualized_results_location)

            cv2.imwrite(image_output_path, new, [cv2.IMWRITE_JPEG_QUALITY, 100])

def parse_args(argv):
    target_folder_path = ''
    if len(sys.argv) < 2:
        raise ValueError ("Missing arguments")
    else:
        target_folder_path = sys.argv[1]

    return target_folder_path

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == '__main__':
    merge_ins = MergeResults()

    input_folder_path = parse_args(sys.argv)

    merge_results = []

    # Merge results obtained from 2 models
    for filename in os.listdir(input_folder_path):
        filepath = os.path.join(input_folder_path, filename)

        if os.path.isfile(filepath) and filename.endswith(".jpg"):
            merge_results.append(merge_ins.merge_results(filename, filepath))

    # Save to specific location
    merge_ins.save_results(input_folder_path, merge_results)

    # Visualize results
    merge_ins.visualize_results()

