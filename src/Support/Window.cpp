#include "Window.h"

bool Window::init(DXContext* contextPtr, int w, int h) {

    width = w;
    height = h;

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
    wcex.lpszClassName = (LPCWSTR)("BreakpointWndCls");
    //wcex.hIconSm = LoadIconW(nullptr, IDI_APPLICATION);
    wndClass = RegisterClassExW(&wcex);
    if (wndClass == 0) {
        return false;
    }

    POINT pos{ 0, 0 };
    GetCursorPos(&pos);
    HMONITOR monitor = MonitorFromPoint(pos, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO monitorInfo{};
    monitorInfo.cbSize = sizeof(monitorInfo);
    GetMonitorInfoW(monitor, &monitorInfo);

    window = CreateWindowExW(WS_EX_OVERLAPPEDWINDOW | WS_EX_APPWINDOW,
        (LPCWSTR)wndClass,
        (LPCWSTR)("Breakpoint"),
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        monitorInfo.rcWork.left + 0,
        monitorInfo.rcWork.top + 0,
        width,
        height,
        nullptr,
        nullptr,
        wcex.hInstance,
        nullptr);

    if (!window) {
        return false;
    }

    //describe swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.Stereo = false; //3d
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0; //no MSAA
    swapChainDesc.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = FRAME_COUNT;
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    DXGI_SWAP_CHAIN_FULLSCREEN_DESC swapChainFullScreenDesc{};
    swapChainFullScreenDesc.Windowed = true;

    //swap chain creation
    dxContext = contextPtr;
    auto& factory = dxContext->getFactory();
    ComPointer<IDXGISwapChain1> swapChain1;
    factory->CreateSwapChainForHwnd(dxContext->getCommandQueue(), window, &swapChainDesc, &swapChainFullScreenDesc, nullptr, &swapChain1);
    if (!swapChain1.QueryInterface(swapChain)) {
        return false;
    }
}

void Window::update() {
    MSG msg;
    while (PeekMessageW(&msg, window, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void Window::present() {
    swapChain->Present(1, 0);
}

void Window::shutdown() {
    swapChain.Release();

    if (window) {
        DestroyWindow(window);
    }

    if (wndClass) {
        UnregisterClassW((LPCWSTR)wndClass, GetModuleHandleW(nullptr));
    }
}

LRESULT Window::OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CLOSE:
        get().shouldClose = true;
        return 0;
    }
    return DefWindowProc(wnd, msg, wParam, lParam);
}
