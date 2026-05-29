"""Minimal YOLOv5 export metadata used by DetectMultiBackend.

The vendored ApexAimBot runtime only needs ``export_formats().Suffix`` to
classify weight file types at inference time. Full model export is intentionally
not vendored here.
"""

from __future__ import annotations

import pandas as pd


def export_formats() -> pd.DataFrame:
    """Return YOLOv5 backend suffix metadata expected by models.common."""
    rows = [
        ("PyTorch", "-", ".pt", True, True),
        ("TorchScript", "torchscript", ".torchscript", True, True),
        ("ONNX", "onnx", ".onnx", True, True),
        ("OpenVINO", "openvino", "_openvino_model", True, False),
        ("TensorRT", "engine", ".engine", False, True),
        ("CoreML", "coreml", ".mlmodel", True, False),
        ("TensorFlow SavedModel", "saved_model", "_saved_model", True, True),
        ("TensorFlow GraphDef", "pb", ".pb", True, True),
        ("TensorFlow Lite", "tflite", ".tflite", True, False),
        ("TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", True, False),
        ("TensorFlow.js", "tfjs", "_web_model", True, False),
        ("PaddlePaddle", "paddle", "_paddle_model", True, True),
    ]
    return pd.DataFrame(rows, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])
