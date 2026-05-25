# OverlayAssist targeting modules

## Copy to your `OverlayAssist` folder

| File | Purpose |
|------|---------|
| **`detector.py`** | Body-structure targeting (one file — not `detection.py`) |
| **`motion.py`** | Smooth tracking when dummy moves (use with detector) |
| **`self_check.py`** | Setup self-check (body dummy test; helpers inlined) |

Optional: `overlay_assist.py`, `setup_doctor.py`

Do **not** replace `aba.py` or `run_windows.bat`.

## In-game: moving dummy, not sky red blobs

Firing-range dummies (your second screenshot) have red on **face, chest, and joints** on a white/grey body — not one solid red blob. The detector must track the **humanoid column**, not whichever red pixel is brightest.

**detector.py** now:
- Merges left/right red plates into one column
- Rejects floating sky/UI blobs
- Aims at **upper chest** on the body bbox (stable while moving)

**motion.py** — in your runtime loop, pass bbox so aim does not jump between red plates:

```python
motion = tracker.observe_target(
    t.centroid_x, t.centroid_y, time.time(),
    bbox_x=t.bbox_x, bbox_y=t.bbox_y, bbox_w=t.bbox_w, bbox_h=t.bbox_h,
)
```

If you only copy `detector.py` without `motion.py` or without wiring `observe_target`, ADS tracking will still feel jumpy.

## Verify setup

```powershell
python setup_doctor.py
python aba.py --self-check
```

Expect: `Detection (synthetic body dummy): OK`
