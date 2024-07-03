cd ../yolact
python eval.py --trained_model=./weights/yolact_resnet50_pipe_378_70000.pth \
    --config=yolact_resnet50_pipe_config \
    --score_threshold=0.8 \
    --top_k=15 \
    --save_mask_info=../script/results/mask_detections.json \
    --images=../script/data/img:../script/results/output_images


