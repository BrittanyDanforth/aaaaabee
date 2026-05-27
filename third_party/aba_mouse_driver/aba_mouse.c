/*
 * aba_mouse.dll — open-source Windows mouse helper for OverlayAssist (ABA).
 *
 * Same export names as ApexAimBot's ghub_mouse.dll API, but implemented with
 * Win32 SendInput only. Build from this source on your machine; do not download
 * unknown DLLs from cheat repos.
 *
 * Build (Windows): scripts\build_aba_mouse_dll.bat
 * Build (cross):   scripts/build_aba_mouse_dll.sh
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static int g_open = 0;

__declspec(dllexport) int mouse_open(void) {
    g_open = 1;
    return 1;
}

__declspec(dllexport) void moveR(int x, int y, int relative) {
    INPUT inp;
    (void)relative; /* Apex always passes TRUE for relative */
    if (!g_open || (x == 0 && y == 0)) {
        return;
    }
    ZeroMemory(&inp, sizeof(inp));
    inp.type = INPUT_MOUSE;
    inp.mi.dx = x;
    inp.mi.dy = y;
    inp.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &inp, sizeof(INPUT));
}

static DWORD down_flag(int code) {
    switch (code) {
    case 1:
        return MOUSEEVENTF_LEFTDOWN;
    case 2:
        return MOUSEEVENTF_MIDDLEDOWN;
    case 3:
        return MOUSEEVENTF_RIGHTDOWN;
    default:
        return 0;
    }
}

static DWORD up_flag(int code) {
    switch (code) {
    case 1:
        return MOUSEEVENTF_LEFTUP;
    case 2:
        return MOUSEEVENTF_MIDDLEUP;
    case 3:
        return MOUSEEVENTF_RIGHTUP;
    default:
        return 0;
    }
}

static void mouse_button(int code, BOOL down) {
    INPUT inp;
    DWORD flag = down ? down_flag(code) : up_flag(code);
    if (!g_open || flag == 0) {
        return;
    }
    ZeroMemory(&inp, sizeof(inp));
    inp.type = INPUT_MOUSE;
    inp.mi.dwFlags = flag;
    SendInput(1, &inp, sizeof(INPUT));
}

__declspec(dllexport) void mouse_down(int code) { mouse_button(code, TRUE); }
__declspec(dllexport) void mouse_up(int code) { mouse_button(code, FALSE); }

__declspec(dllexport) void scroll(int delta) {
    INPUT inp;
    if (!g_open) {
        return;
    }
    ZeroMemory(&inp, sizeof(inp));
    inp.type = INPUT_MOUSE;
    inp.mi.dwFlags = MOUSEEVENTF_WHEEL;
    inp.mi.mouseData = (DWORD)delta;
    SendInput(1, &inp, sizeof(INPUT));
}

__declspec(dllexport) void key_down(int code) {
    INPUT inp;
    if (!g_open) {
        return;
    }
    ZeroMemory(&inp, sizeof(inp));
    inp.type = INPUT_KEYBOARD;
    inp.ki.wVk = (WORD)(code & 0xFF);
    inp.ki.dwFlags = 0;
    SendInput(1, &inp, sizeof(INPUT));
}

__declspec(dllexport) void key_up(int code) {
    INPUT inp;
    if (!g_open) {
        return;
    }
    ZeroMemory(&inp, sizeof(inp));
    inp.type = INPUT_KEYBOARD;
    inp.ki.wVk = (WORD)(code & 0xFF);
    inp.ki.dwFlags = KEYEVENTF_KEYUP;
    SendInput(1, &inp, sizeof(INPUT));
}

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst;
    (void)reserved;
    if (reason == DLL_PROCESS_DETACH) {
        g_open = 0;
    }
    return TRUE;
}
