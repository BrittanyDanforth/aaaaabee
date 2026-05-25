# Real Apex test assets

## GIF combat recording (166 frames)

| Path | Description |
|------|-------------|
| `_gif_frames/source.gif` | Original recording |
| `_gif_frames/frame_*.png` | Sparse samples (every 5 frames) |
| `_gif_frames_all/` | All 166 extracted frames (run `scripts/extract_gif_frames.py`) |
| **`gif_166_proof/`** | **Latest audit** — red dot on real frames, summary, montage |

### Regenerate proof artifacts

```bash
python3 scripts/extract_gif_frames.py
python3 scripts/audit_gif_full_sequence.py --save-every 10
```

Open `gif_166_proof/proof_montage.jpg` for a quick visual pass. Per-frame PNGs with **red dot** are in `gif_166_proof/frames/`.

## Static screenshots (before/after)

`before/` and `after/` hold detector audit outputs (`meta.json`, masks) from `audit_real_apex.py` when `_inputs/` screenshots are present.
