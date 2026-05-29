# ApexAimBot `function/weights/` — full catalog

Upstream folder is **~120 MB** (many games). ABA only ships **Apex** files below.

## Bundled in this repo (SHA-256 verified)

| File | SHA-256 | Size | Use |
|------|---------|------|-----|
| `APEX416SFP32.engine` | `e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16` | 7.9 MiB | Default in `1bit.ai.config` — TensorRT @ 416 |
| `APEX22W.pt` | `fea589b8d3d6097d1dd66659fb68dc8c90e23de93a27627458742cd9707aa144` | 3.7 MiB | PyTorch fallback if TensorRT unavailable |

Set in config:

```json
"yolo_weights_path": "third_party/apexaimbot/weights/APEX416SFP32.engine"
```

or `APEX22W.pt` for CPU/.pt path.

## Upstream only — do NOT bundle (other games)

| File | ~Size | Game |
|------|-------|------|
| PUBG22WBODY.engine / .onnx / .pt | 14–31 MB | PUBG |
| PUBG模型2W图body单分类.pt | 14 MB | PUBG |
| csgo_5n_48000.pt | 15 MB | CS:GO |
| 瓦 紫色盲 标识0.pt | 14 MB | Valorant |
| yolov5s.pt | 15 MB | COCO baseline |
| A.onnx, BJGFP32.onnx | 7 MB | Other |

`scripts/ensure_apexaimbot_weights.py` **refuses** to copy these (wrong SHA / not in allowlist) and flags any stray file under `weights/`.

## Stale name in upstream `main.py`

`main.py` default `weight = 'APEX416.engine'` does **not** exist in the repo. Real file is **`APEX416SFP32.engine`** per `1bit.ai.config`.

Verify: `python3 scripts/ensure_apexaimbot_weights.py` (or `ensure_apexaimbot_bundle.py` for vendor tree too)
