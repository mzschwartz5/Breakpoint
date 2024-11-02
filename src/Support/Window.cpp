#include "Window.h"

bool Window::Init() {

	WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(wcex);
    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = &Window::OnWindowMessage;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = GetModuleHandle(nullptr);
    //wcex.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    //wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = L"D3D12ExWndCls";
    //wcex.hIconSm = LoadIconW(nullptr, IDI_APPLICATION);
    m_wndClass = RegisterClassExW(&wcex);
    if (m_wndClass == 0) {
        return false;
    }

    POINT pos{ 0, 0 };
    GetCursorPos(&pos);
    HMONITOR monitor = MonitorFromPoint(pos, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO monitorInfo{};
    monitorInfo.cbSize = sizeof(monitorInfo);
    GetMonitorInfoW(monitor, &monitorInfo);

    m_window = CreateWindowExW(WS_EX_OVERLAPPEDWINDOW | WS_EX_APPWINDOW, 
                               (LPCWSTR)m_wndClass, 
                               L"Breakpoint", 
                               WS_OVERLAPPEDWINDOW | WS_VISIBLE, 
                               monitorInfo.rcWork.left + 0, 
                               monitorInfo.rcWork.top + 0, 
                               1920, 
                               1080, 
                               nullptr, 
                               nullptr, 
                               wcex.hInstance, 
                               nullptr);

	return m_window != nullptr;

}

void Window::Update() {
    MSG msg;
    while (PeekMessageW(&msg, m_window, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void Window::Shutdown() {
    if (m_window) {
        DestroyWindow(m_window);
    }

    if (m_wndClass) {
        UnregisterClassW((LPCWSTR)m_wndClass, GetModuleHandleW(nullptr));
    }
}

LRESULT Window::OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CLOSE:
        Get().m_shouldClose = true;
        return 0;
    }
    return DefWindowProc(wnd, msg, wParam, lParam);
}
