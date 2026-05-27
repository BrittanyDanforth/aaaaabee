# YOLO weights

Place Apex (or custom) YOLOv5 weights here for `detection_mode: yolo`.

Example: copy `APEX416.pt` or TensorRT `.engine` from [ApexAimBot](https://github.com/1bit-monster7/ApexAimBot) and set:

```json
"yolo_weights_path": "models/apex_yolo.pt"
```

For `.engine` / `.onnx`, also set `yolo_yolov5_root` to their `yolov5` folder.
