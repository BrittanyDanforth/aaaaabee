# Why You See Two Rings (And Why Only One Is Ours)

**TL;DR:** OverlayAssist draws exactly one green ring — the outer green oval.
The inner ring you see during ADS is the game's own optic reticle (1× HCOG /
red-dot / Bruiser / etc.). That inner ring is rendered by Apex Legends itself;
no code in this project ever draws it, and disabling our overlay does NOT make
it disappear.

This brief documents the empirical evidence, links every relevant source
location, and tells you how to verify the distinction in-game.

---

## 1. The overlay code only ever creates ONE oval

`overlay_window.py:342` is the only `create_oval` call in the entire overlay
build path. The single oval is tagged `fov_ring`:

```342:350:overlay_window.py
        self._fov_id = self._canvas.create_oval(
            self._cx - fov_radius,
            self._cy - fov_radius,
            self._cx + fov_radius,
            self._cy + fov_radius,
            outline="#00ff88",
            width=2,
            tags=("fov_ring",),
        )
```

That is the entire FOV ring construction. No other `create_oval` call exists
that uses any green outline or anything that could plausibly be confused for
an FOV indicator. The crosshair (`create_line` at `:353`/`:362`) and the
target dot (`create_oval` at `:375` initialised with empty outline AND
`state="hidden"`) cannot present as a green ring.

## 2. The redraw loop actively purges any second `fov_ring` item

`overlay_window.py:424-429` runs on every redraw and destroys any canvas item
that ever ends up tagged `fov_ring` but isn't the one we created:

```424:429:overlay_window.py
            stray = [
                item for item in self._canvas.find_withtag("fov_ring")
                if item != self._fov_id
            ]
            for item in stray:
                self._canvas.delete(item)
```

If a second oval ever appeared on the canvas with that tag, this loop would
delete it before the next paint. So even bugs that tried to spawn a ghost ring
on the canvas would self-heal within one frame.

## 3. A regression test pins the invariant

`tests/test_overlay_single_ring.py` is the dedicated single-ring regression
test. It loads the overlay's `_redraw` path and asserts that the canvas
contains exactly one item with the `fov_ring` tag after each redraw,
including the case where a malicious caller has tried to inject a second
oval. Companion test `tests/test_overlay_one_visible_ring.py` asserts that
`debug_show_detect_ring` defaults to False so the optional second ring on the
OpenCV debug window (NOT the on-screen overlay) only fires when explicitly
enabled.

## 4. What you're actually seeing in the screenshot

When you ADS with a 1× HCOG / red-dot / Bruiser optic, the optic itself draws
a reticle on the in-game viewmodel. That reticle includes a circular ring (or
chevron, depending on the optic). The game renders that as part of the
weapon-view 3-D pipeline — it is BAKED INTO the screen pixels Apex paints
before our overlay window even gets a chance to composite the green FOV ring
on top.

Our overlay's green oval sits at the outside (it tracks the detection FOV
radius, ~36-42 % of the smaller monitor axis). The optic reticle sits at the
inside (it tracks the iron-sight crosshair, typically ~20-50 px from screen
center on 1080p). Their concentric appearance is purely a visual coincidence
of two unrelated drawings.

## 5. How to confirm with one minute of in-game testing

There is no code change needed; this is a settings-only verification:

1. Drop into the firing range with **iron sights** (a weapon equipped with NO
   optic — base P2020 / RE-45 / Mozambique, or strip the optic off any rifle
   in the loadout menu).
2. ADS. Observe: only the OUTER green ring is visible. There is no inner
   ring at all because the iron sights have no reticle of their own.
3. Now swap to a **1× HCOG** / **Bruiser** / **HCOG Classic** / **digital
   threat**. ADS. Observe: the optic's circle / chevron appears INSIDE our
   outer green ring. Our outer ring did not move, did not grow a child, did
   not split — the optic simply painted its reticle into the scene.
4. Bonus: pull the optic off the weapon mid-ADS via the inventory. You will
   watch the inner ring vanish in real time while the outer green ring sits
   unchanged.

If at any point in step 2 you still see two green rings, that's an actual
bug worth investigating — but no piece of evidence we have seen so far
matches that condition. Every report has involved an optic being equipped.

## 6. Where this is documented in code comments

`overlay_window.py:337-341` (immediately above the single create_oval) and
`overlay_window.py:418-423` (immediately above the purge) both call out the
"ghost ring" failure mode by name. The intent of the tag + purge + regression
test is to make the single-ring invariant impossible to violate from inside
this codebase. Any inner ring the user sees has to be coming from somewhere
the codebase does not control — and that's the game's own optic system.

## 7. Files cited

- `overlay_window.py:342` — single `create_oval` for the FOV ring.
- `overlay_window.py:424-429` — orphan-ring purge in `_redraw`.
- `tests/test_overlay_single_ring.py` — single-ring regression test.
- `tests/test_overlay_one_visible_ring.py` — `debug_show_detect_ring` default.

No code change is required to address the user's two-rings report. The inner
ring is the game's optic reticle.
