#!/bin/bash

cd ../coordinate-transform/

./build/Rotation \
--ply ../res_b2b/data/ply/Frame_03.ply \
--img ../res_b2b/data/img/PIC_06181.jpg \
--dir ../res_b2b/out/ \
--mode 0

cd ../detect/
rye run python detect_by_yolov5.py ../res_b2b/data/img ../res_b2b/out/detect
# rye run python detect_by_yolact.py ../res_b2b/data/img ../res_b2b/out/detect
rye run python merge_results.py ../res_b2b/data/img ../res_b2b/out/detect




cd ../coordinate-transform/
./build/Rotation \
--ply ../res_b2b/out/mse_lidar.ply \
--img ../res_b2b/data/img/PIC_06181.jpg \
--json ../res_b2b/out/detect/detections.json \
--dir ../res_b2b/out/ \
--mode 6
