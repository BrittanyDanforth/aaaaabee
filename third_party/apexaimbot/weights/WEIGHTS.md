# Bundled detection weights

| File | SHA-256 | Size | Source |
|------|---------|------|--------|
| `APEX416SFP32.engine` | `e8d61fede4f6f59fb24cf1245394013e52d21a0a3a374cac39085db1c6e6dd16` | 7.9 MiB | [ApexAimBot](https://github.com/1bit-monster7/ApexAimBot) `function/weights/` |

This is a **TensorRT engine binary** (not an executable). ABA loads it only via PyTorch `DetectMultiBackend` inside the vendored YOLOv5 code.

Verify after clone:

```bash
python3 scripts/ensure_apexaimbot_weights.py
```

If the hash fails, restore from the upstream repo or re-run the script with `APEXAIMBOT_WEIGHTS_SRC` pointing at a local ApexAimBot clone.
