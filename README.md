# detect
研究で使用した物体検出まわりのscript.
yolov5と yolactを動かし、推論を行ない、結果をlab-pointcloudのprogram用に変換。


## dependencies
pyproject.tomlに記載しているdependenciesをインストールしてください。
```toml
dependencies = [
    "gitpython>=3.1.43",
    "matplotlib>=3.9.0",
    "opencv-python>=4.10.0.84",
    "pillow>=10.4.0",
    "psutil>=6.0.0",
    "pyyaml>=6.0.1",
    "requests>=2.32.3",
    "scipy>=1.14.0",
    "thop>=0.1.1.post2209072238",
    "torch>=2.3.1",
    "torchvision>=0.18.1",
    "tqdm>=4.66.4",
    "ultralytics>=8.2.48",
    "numpy>=1.23.5",
    "pandas>=2.2.2",
    "seaborn>=0.13.2",
    "setuptools>=70.2.0",
    "wheel>=0.43.0",
    "pycocotools>=2.0.8",
]
```

NOTE:  
detectと同じ階層にyolov5, yolactをcloneしてください。
変える場合は、python_scriptのの中身を変更してください。




## Usage
- `detect_by_yolov5.py`は、検出にかける画像のディレクトリパスと、検出結果であるjsonとbboxを可視化したものを保存するディレクトリを指定してください。  
bbox_detections.jsonが出力されます。

```bash
detect_by_yolov5.py [検出タイショウの画像DIRパス] [出力DIRパス]

python detect_by_yolov5.py../res/data/img ../res/out/detect
```

- `merge_results.py`は、yolov5とyolactの結果を統合したjsonファイルを出力します。  
検出にかけた画像のディレクトリパスと、  
yolov5とyolactの結果`bbox_detections.json`, `mask_detections.json`が格納されているディレクトリを指定してください。  
統合結果である`detections.json`が出力されます。
```bash
merge_results.py [検出対象の画像DIRパス] [yolov5, yolactの結果が格納されいている


```

- `merge_truth_detect_bbox.py`は、おそらくbboxとtruthの結果を統合して、画像で比較するためのものだったんだろうと推測していますが、なさそうです。
