"""Optional fullscreen transparent overlay (FOV ring + target dot). Linux/X11 or Windows."""

from __future__ import annotations

import logging
import sys
import threading
import tkinter as tk

logger = logging.getLogger("overlay_assist")

_TRANSPARENT_BG = "#010203"


def _hex_to_colorref(hex_color: str) -> int:
    """Convert #RRGGBB to Win32 COLORREF 0x00bbggrr."""
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return r | (g << 8) | (b << 16)


def _collect_overlay_hwnds(root: tk.Tk, canvas: tk.Canvas | None) -> list[int]:
    """Return every native window handle that can steal mouse input from the game."""
    hwnds: list[int] = []
    seen: set[int] = set()

    def _add(hwnd: int | None) -> None:
        if hwnd and hwnd not in seen:
            seen.add(hwnd)
            hwnds.append(hwnd)

    root_id = root.winfo_id()
    _add(root_id)

    if canvas is not None:
        _add(canvas.winfo_id())

    if sys.platform != "win32":
        return hwnds

    try:
        import ctypes

        if not hasattr(ctypes, "WINFUNCTYPE"):
            return hwnds

        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        _add(user32.GetParent(root_id))

        if canvas is not None:
            _add(user32.GetParent(canvas.winfo_id()))

        def _enum_child(child_hwnd, _lparam):
            _add(child_hwnd)
            return True

        enum_proc = WNDENUMPROC(_enum_child)
        user32.EnumChildWindows(root_id, enum_proc, 0)
        _ = enum_proc
    except Exception:
        logger.debug("Could not enumerate overlay HWNDs", exc_info=True)

    return hwnds


def _apply_win32_passive_hwnd(hwnd: int, *, colorkey: int) -> None:
    """Apply layered, click-through, no-activate styles to one HWND."""
    import ctypes

    user32 = ctypes.windll.user32

    is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
    LONG_PTR = ctypes.c_longlong if is_64 else ctypes.c_long
    UINT = ctypes.c_uint
    HWND = ctypes.c_void_p

    GWL_STYLE = -16
    GWL_EXSTYLE = -20

    WS_DISABLED = 0x08000000

    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_NOACTIVATE = 0x08000000
    WS_EX_TOOLWINDOW = 0x00000080

    LWA_COLORKEY = 0x00000001

    if hasattr(user32, "GetWindowLongPtrW"):
        get_window_long = user32.GetWindowLongPtrW
        set_window_long = user32.SetWindowLongPtrW
    else:
        get_window_long = user32.GetWindowLongW
        set_window_long = user32.SetWindowLongW

    get_window_long.argtypes = [HWND, ctypes.c_int]
    get_window_long.restype = LONG_PTR
    set_window_long.argtypes = [HWND, ctypes.c_int, LONG_PTR]
    set_window_long.restype = LONG_PTR

    user32.SetLayeredWindowAttributes.argtypes = [
        HWND,
        ctypes.c_uint32,
        ctypes.c_ubyte,
        ctypes.c_uint32,
    ]
    user32.SetLayeredWindowAttributes.restype = ctypes.c_bool

    user32.EnableWindow.argtypes = [HWND, ctypes.c_bool]
    user32.EnableWindow.restype = ctypes.c_bool

    ex_style = get_window_long(hwnd, GWL_EXSTYLE)
    ex_style = (
        ex_style
        | WS_EX_LAYERED
        | WS_EX_TRANSPARENT
        | WS_EX_NOACTIVATE
        | WS_EX_TOOLWINDOW
    )
    set_window_long(hwnd, GWL_EXSTYLE, ex_style)

    style = get_window_long(hwnd, GWL_STYLE)
    set_window_long(hwnd, GWL_STYLE, style | WS_DISABLED)

    user32.SetLayeredWindowAttributes(hwnd, colorkey, 0, LWA_COLORKEY)
    user32.EnableWindow(hwnd, False)


def _apply_x11_click_through(root: tk.Tk) -> None:
    """Make the overlay ignore pointer events on X11 (Linux)."""
    try:
        from tkinter import _tkinter

        display_name = root.winfo_screen()._display.__str__()
    except Exception:
        display_name = None

    try:
        import ctypes
        import ctypes.util

        x11 = ctypes.CDLL(ctypes.util.find_library("X11"))
        if not x11:
            return

        class XRectangle(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_short),
                ("y", ctypes.c_short),
                ("width", ctypes.c_ushort),
                ("height", ctypes.c_ushort),
            ]

        fixes = ctypes.CDLL(ctypes.util.find_library("Xfixes"))
        if not fixes:
            return

        display = x11.XOpenDisplay(display_name.encode() if display_name else None)
        if not display:
            return

        try:
            window = root.winfo_id()
            fixes.XFixesSetWindowShapeRegion(display, window, 1, 0, 0, None)  # 1 = ShapeInput
        finally:
            x11.XCloseDisplay(display)
    except Exception:
        logger.debug("Could not set X11 input shape for overlay", exc_info=True)


def _make_click_through(root: tk.Tk, canvas: tk.Canvas | None = None) -> None:
    """Set platform styles so the overlay is visible but never receives mouse/focus input."""
    canvas = canvas or getattr(root, "_overlay_canvas", None)

    try:
        root.attributes("-disabled", True)
    except tk.TclError:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            HWND = ctypes.c_void_p
            UINT = ctypes.c_uint

            SW_SHOWNOACTIVATE = 4
            HWND_TOPMOST = -1
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020
            SWP_SHOWWINDOW = 0x0040

            colorkey = _hex_to_colorref(_TRANSPARENT_BG)
            hwnds = _collect_overlay_hwnds(root, canvas)

            for hwnd in hwnds:
                _apply_win32_passive_hwnd(hwnd, colorkey=colorkey)

            if hwnds:
                top = hwnds[0]
                user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
                user32.ShowWindow.restype = ctypes.c_bool
                user32.SetWindowPos.argtypes = [
                    HWND,
                    HWND,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    UINT,
                ]
                user32.SetWindowPos.restype = ctypes.c_bool

                user32.ShowWindow(top, SW_SHOWNOACTIVATE)
                user32.SetWindowPos(
                    top,
                    HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
                )
                for hwnd in hwnds:
                    user32.EnableWindow(hwnd, False)
        except Exception:
            logger.debug("Could not set passive click-through overlay styles", exc_info=True)
        return

    if sys.platform.startswith("linux"):
        _apply_x11_click_through(root)


class OverlayWindow:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        fov_radius: int,
        origin_x: int = 0,
        origin_y: int = 0,
        *,
        fov_center_x: float | None = None,
        fov_center_y: float | None = None,
    ) -> None:
        self._fov_radius = fov_radius
        self._drawn_fov_radius = fov_radius
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._fov_center_x = fov_center_x
        self._fov_center_y = fov_center_y
        self._target: tuple[float, float] | None = None
        self._active = False
        self._closed = False
        self._lock = threading.Lock()
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._fov_id = None
        self._cross_h = None
        self._cross_v = None
        self._target_id = None
        self._status_id = None
        self._cx = 0
        self._cy = 0

    def _build_ui(self) -> None:
        sw = self._screen_width
        sh = self._screen_height
        fov_radius = self._fov_radius

        # DPI awareness must be set BEFORE this thread creates its Tk
        # interpreter, otherwise Tk samples the wrong system DPI and the
        # canvas reports logical pixels while mss feeds physical pixels.
        # Re-asserting here is idempotent at the OS level but keeps the
        # overlay thread honest if it was spawned before main-thread setup.
        try:
            from platform_info import enable_dpi_awareness

            enable_dpi_awareness()
        except Exception:
            logger.debug("Could not assert DPI awareness on overlay thread", exc_info=True)

        self._root = tk.Tk()
        self._root.title("OverlayAssist")
        # Pin Tk's internal scale factor to 1.0 so canvas pixel coords match
        # mss physical pixels. Without this, Tcl uses fpixels/screen DPI to
        # scale fonts/some shapes and the FOV ring drifts off the crosshair.
        try:
            self._root.tk.call("tk", "scaling", 1.0)
        except tk.TclError:
            logger.debug("Could not pin Tk scaling to 1.0", exc_info=True)
        # overrideredirect MUST be set before geometry on Windows — otherwise
        # the geometry +x+y positions the outer (decorated) frame and the
        # subsequent decoration removal shifts the client area UP by the
        # title-bar height, leaving the FOV ring ~30px above the crosshair.
        self._root.overrideredirect(True)
        self._root.geometry(f"{sw}x{sh}+{self._origin_x}+{self._origin_y}")
        self._root.attributes("-topmost", True)

        try:
            self._root.config(bg=_TRANSPARENT_BG)
            self._root.attributes("-transparentcolor", _TRANSPARENT_BG)
        except tk.TclError:
            logger.warning("Transparent overlay background is not supported on this system")
            try:
                self._root.destroy()
            except tk.TclError:
                pass
            raise RuntimeError("Transparent overlay background is not supported")

        self._canvas = tk.Canvas(
            self._root,
            width=sw,
            height=sh,
            highlightthickness=0,
            bd=0,
            bg=_TRANSPARENT_BG,
            takefocus=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._root._overlay_canvas = self._canvas

        fcx = self._fov_center_x
        fcy = self._fov_center_y
        self._cx = int(round(fcx if fcx is not None else sw / 2))
        self._cy = int(round(fcy if fcy is not None else sh / 2))

        self._fov_id = self._canvas.create_oval(
            self._cx - fov_radius,
            self._cy - fov_radius,
            self._cx + fov_radius,
            self._cy + fov_radius,
            outline="#00ff88",
            width=2,
        )
        self._drawn_fov_radius = fov_radius

        self._cross_h = self._canvas.create_line(
            self._cx - 10,
            self._cy,
            self._cx + 10,
            self._cy,
            fill="#ffffff",
            width=1,
        )

        self._cross_v = self._canvas.create_line(
            self._cx,
            self._cy - 10,
            self._cx,
            self._cy + 10,
            fill="#ffffff",
            width=1,
        )

        self._target_id = self._canvas.create_oval(0, 0, 0, 0, outline="", fill="")

        self._status_id = self._canvas.create_text(
            12,
            12,
            anchor="nw",
            text="OverlayAssist",
            fill="#aaaaaa",
            font=("DejaVu Sans", 10),
        )

        self._root.update_idletasks()
        self._root.update()

        def _reapply() -> None:
            if self._root is not None:
                _make_click_through(self._root, self._canvas)

        _reapply()
        for delay in (25, 100, 300, 1000):
            self._root.after(delay, _reapply)

    def set_fov_radius(self, radius: int) -> None:
        with self._lock:
            self._fov_radius = max(40, int(radius))

    def set_state(self, ads: bool, target: tuple[float, float] | None) -> None:
        with self._lock:
            self._active = ads
            self._target = target

    def _redraw(self) -> None:
        if self._canvas is None or self._fov_id is None:
            return

        with self._lock:
            ads = self._active
            target = self._target
            fov_radius = self._fov_radius

        try:
            color = "#00ff88" if ads else "#446644"
            self._canvas.itemconfig(self._fov_id, outline=color)
            # set_fov_radius() only stores the new value; the canvas oval
            # must be re-coordinated here or the ring stays frozen at its
            # build-time radius (visible ring vs. clamp radius would diverge).
            if fov_radius != self._drawn_fov_radius:
                self._canvas.coords(
                    self._fov_id,
                    self._cx - fov_radius,
                    self._cy - fov_radius,
                    self._cx + fov_radius,
                    self._cy + fov_radius,
                )
                self._drawn_fov_radius = fov_radius

            if target is not None:
                tx, ty = int(round(target[0])), int(round(target[1]))
                r = 6
                self._canvas.coords(self._target_id, tx - r, ty - r, tx + r, ty + r)
                self._canvas.itemconfig(self._target_id, outline="#ff4444", fill="#ff4444")
            else:
                self._canvas.coords(self._target_id, 0, 0, 0, 0)
                self._canvas.itemconfig(self._target_id, outline="", fill="")
        except (tk.TclError, RuntimeError):
            pass

    def run(self) -> None:
        self._closed = False

        try:
            self._build_ui()
        except Exception:
            logger.exception("Overlay window failed to initialize")
            return

        tick_count = 0

        def tick() -> None:
            nonlocal tick_count
            if getattr(self, "_closed", False):
                return

            self._redraw()

            if self._root is not None:
                # Re-apply periodically; Tk can recreate or re-enable child HWNDs.
                tick_count += 1
                if tick_count == 1 or tick_count % 15 == 0:
                    _make_click_through(self._root, self._canvas)
                self._root.after(33, tick)

        tick()

        try:
            assert self._root is not None
            self._root.mainloop()
        except RuntimeError:
            logger.debug("Overlay mainloop exited (expected on secondary thread)")

    def close(self) -> None:
        self._closed = True
        root = self._root

        if root is None:
            return

        def _do_close() -> None:
            try:
                if sys.platform == "win32":
                    try:
                        import ctypes

                        user32 = ctypes.windll.user32
                        for hwnd in _collect_overlay_hwnds(root, self._canvas):
                            user32.EnableWindow(hwnd, True)
                    except Exception:
                        pass

                root.quit()
                root.destroy()
            except tk.TclError:
                pass

        try:
            root.after(0, _do_close)
        except (tk.TclError, RuntimeError):
            pass
