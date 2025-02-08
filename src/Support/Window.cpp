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
    wcex.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = L"BreakpointWndCls";
    wcex.hIconSm = LoadIconW(nullptr, IDI_APPLICATION);
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
        L"Breakpoint",
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

    // Create RTV Heap
    D3D12_DESCRIPTOR_HEAP_DESC descHeapDesc{};
    descHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    descHeapDesc.NumDescriptors = FRAME_COUNT + 2; // +2 for object scene render targets
    descHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    descHeapDesc.NodeMask = 0;
    if (FAILED(dxContext->getDevice()->CreateDescriptorHeap(&descHeapDesc, IID_PPV_ARGS(&rtvDescHeap)))) {
        return false;
    }

    // Create handles to view
    auto firstHandle = rtvDescHeap->GetCPUDescriptorHandleForHeapStart();
    auto handleIncrement = dxContext->getDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    for (size_t i = 0; i < FRAME_COUNT; i++) {
        rtvHandles[i] = firstHandle;
        rtvHandles[i].ptr += handleIncrement * i;
    }

    // Create DSV Heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    if (FAILED(dxContext->getDevice()->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvDescHeap)))) {
        return false;
    }

    // Create handles to view
    dsvHandle = dsvDescHeap->GetCPUDescriptorHandleForHeapStart();

    // Create the depth stencil view
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;
 
    const CD3DX12_HEAP_PROPERTIES depthStencilHeapProps(D3D12_HEAP_TYPE_DEFAULT);
    const CD3DX12_RESOURCE_DESC depthStencilTextureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, width, height, 1, 0, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

    if (FAILED(dxContext->getDevice()->CreateCommittedResource(
        &depthStencilHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &depthStencilTextureDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &depthOptimizedClearValue,
        IID_PPV_ARGS(&depthStencilBuffer)
    ))) {
        return false;
    }

    dxContext->getDevice()->CreateDepthStencilView(depthStencilBuffer, &depthStencilDesc, dsvHandle);
    
    //get buffers
    if (!getBuffers()) {
        return false;
    }

    return true;
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

void Window::resize() {
    releaseBuffers();

    RECT rect;
    if (GetClientRect(window, &rect)) {
        width = rect.right - rect.left;
        height = rect.bottom - rect.top;

        //unknown keeps old format
        swapChain->ResizeBuffers(FRAME_COUNT, width, height, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);
        shouldResize = false;
    }

    getBuffers();
}

void Window::beginFrame(ID3D12GraphicsCommandList6* cmdList) {
    currentSwapChainBufferIdx = swapChain->GetCurrentBackBufferIndex();
    transitionSwapChain(cmdList, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

    float clearColor[] = { 0.9f, 0.9f, 0.9f, 1.f };
    cmdList->ClearRenderTargetView(rtvHandles[currentSwapChainBufferIdx], clearColor, 0, nullptr);
    cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);

    transitionObjectRTs(cmdList, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    
    float clearColorObject[] = { 0.f, 0.f, 0.f, 1.f };
    cmdList->ClearRenderTargetView(objectSceneRTVHandleColor, clearColorObject, 0, nullptr);
    cmdList->ClearRenderTargetView(objectSceneRTVHandlePosition, clearColorObject, 0, nullptr);
    cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.f, 0, 0, nullptr);
}

void Window::setMainRT(ID3D12GraphicsCommandList6* cmdList) {
    cmdList->OMSetRenderTargets(1, &rtvHandles[currentSwapChainBufferIdx], false, &dsvHandle);
}

void Window::setTextureRTs(ID3D12GraphicsCommandList6* cmdList) {
    // Sets 2 render targets for storing the object scene's color and position,
    // and the main render target for the window.
    D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandles[3] = { objectSceneRTVHandleColor, objectSceneRTVHandlePosition, rtvHandles[currentSwapChainBufferIdx] };
    cmdList->OMSetRenderTargets(3, renderTargetHandles, false, &dsvHandle);
}

void Window::transitionSwapChain(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier;
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = swapChainBuffers[currentSwapChainBufferIdx];
    barrier.Transition.Subresource = 0;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;

    cmdList->ResourceBarrier(1, &barrier);
}

void Window::transitionObjectRTs(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    CD3DX12_RESOURCE_BARRIER colorBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        objectSceneColorTexture.Get(),
        before,
        after
    );

    CD3DX12_RESOURCE_BARRIER positionBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        objectScenePositionTexture.Get(),
        before,
        after
    );

    D3D12_RESOURCE_BARRIER barriers[2] = { colorBarrier, positionBarrier };
    cmdList->ResourceBarrier(2, barriers);
}

void Window::shutdown() {
    releaseBuffers();

    rtvDescHeap.Release();

    swapChain.Release();

    if (window) {
        DestroyWindow(window);
    }

    if (wndClass) {
        UnregisterClassW((LPCWSTR)wndClass, GetModuleHandleW(nullptr));
    }
}

void Window::updateTitle(std::wstring text) {
    SetWindowTextW(window, text.c_str());
}

bool Window::getBuffers() {
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        if (FAILED(swapChain->GetBuffer(i, IID_PPV_ARGS(&swapChainBuffers[i])))) {
            return false;
        }

        D3D12_RENDER_TARGET_VIEW_DESC rtv{};
        rtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        rtv.Texture2D.MipSlice = 0;
        rtv.Texture2D.PlaneSlice = 0;
        dxContext->getDevice()->CreateRenderTargetView(swapChainBuffers[i], &rtv, rtvHandles[i]);
    }
    return true;
}

void Window::releaseBuffers() {
    for (size_t i = 0; i < FRAME_COUNT; i++) {
        swapChainBuffers[i].Release();
    }
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT Window::OnWindowMessage(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(wnd, msg, wParam, lParam))
        return true;

    switch (msg) {
        case WM_SIZE:
            //only resize if size is not 0 and size has been changed from expected size
            if (lParam && (LOWORD(lParam) != get().width || HIWORD(lParam) != get().height)) {
                get().shouldResize = true;
            }
            break;
        case WM_CLOSE:
            get().shouldClose = true;
            return 0;
        case WM_ACTIVATEAPP:
            DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
            DirectX::Mouse::ProcessMessage(msg, wParam, lParam);
            break;
        case WM_ACTIVATE:
        case WM_INPUT:
        case WM_MOUSEMOVE:
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_MOUSEWHEEL:
        case WM_MOUSEHOVER:
            DirectX::Mouse::ProcessMessage(msg, wParam, lParam);
            break;
        case WM_MOUSEACTIVATE:
            //ignore first click when returning to window
            return MA_ACTIVATEANDEAT;
        case WM_XBUTTONDOWN:
        case WM_XBUTTONUP:
        case WM_KEYDOWN:
        case WM_KEYUP:
        case WM_SYSKEYUP:
            DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
            break;
        case WM_SYSKEYDOWN:
            DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
            if (wParam == VK_RETURN && (lParam & 0x60000000) == 0x20000000) {}
            break;
        case WM_CHAR:
            switch (wParam)
                case VK_ESCAPE:
                    get().shouldClose = true;
                    DirectX::Keyboard::ProcessMessage(msg, wParam, lParam);
                    return 0;
    }
    return DefWindowProc(wnd, msg, wParam, lParam);
}


void Window::createViewport(D3D12_VIEWPORT& vp) {
    vp.TopLeftX = vp.TopLeftY = 0;
    vp.Width = (float)Window::get().getWidth();
    vp.Height = (float)Window::get().getHeight();
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
}

void Window::setViewport(D3D12_VIEWPORT& vp, ID3D12GraphicsCommandList5* cmdList) {
    cmdList->RSSetViewports(1, &vp);
    RECT scRect;
    scRect.left = scRect.top = 0;
    scRect.right = Window::get().getWidth();
    scRect.bottom = Window::get().getHeight();
    cmdList->RSSetScissorRects(1, &scRect);
}

bool Window::createObjectSceneRenderTargets(DescriptorHeap* srvDescHeap) {
    // Describe the off-screen render target texture for color
    D3D12_RESOURCE_DESC textureDesc = {};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    clearValue.Color[0] = 0.0f;
    clearValue.Color[1] = 0.0f;
    clearValue.Color[2] = 0.0f;
    clearValue.Color[3] = 1.0f;

    // Create the off-screen render target texture for color
    CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);
    if (FAILED(dxContext->getDevice()->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        &clearValue,
        IID_PPV_ARGS(&objectSceneColorTexture)))) {
        return false;
    }

    objectSceneColorTexture->SetName(L"Object Scene Color Texture");

    // Create RTV for the off-screen render target for color 
    D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
    rtvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    rtvDesc.Texture2D.MipSlice = 0;
    auto rtvHandleIncrement = dxContext->getDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    objectSceneRTVHandleColor = rtvDescHeap->GetCPUDescriptorHandleForHeapStart();
    objectSceneRTVHandleColor.ptr += rtvHandleIncrement * FRAME_COUNT;
    dxContext->getDevice()->CreateRenderTargetView(objectSceneColorTexture.Get(), &rtvDesc, objectSceneRTVHandleColor);

    // Create SRV for the off-screen render target for color
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

    unsigned int nextSrvIndex = srvDescHeap->GetNextAvailableIndex();
    D3D12_CPU_DESCRIPTOR_HANDLE objectSceneSRVCPUColor = srvDescHeap->GetCPUHandleAt(nextSrvIndex);
    dxContext->getDevice()->CreateShaderResourceView(objectSceneColorTexture.Get(), &srvDesc, objectSceneSRVCPUColor);
    objectSceneSRVHandleColor = srvDescHeap->GetGPUHandleAt(nextSrvIndex);

    // Repeat the above steps for the position render target
    textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    clearValue.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; 
    if (FAILED(dxContext->getDevice()->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        &clearValue,
        IID_PPV_ARGS(&objectScenePositionTexture)))) {
        return false;
    }

    objectScenePositionTexture->SetName(L"Object Scene Position Texture");

    rtvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    objectSceneRTVHandlePosition = rtvDescHeap->GetCPUDescriptorHandleForHeapStart();
    objectSceneRTVHandlePosition.ptr += dxContext->getDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) * (FRAME_COUNT + 1);
    dxContext->getDevice()->CreateRenderTargetView(objectScenePositionTexture.Get(), &rtvDesc, objectSceneRTVHandlePosition);

    srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    nextSrvIndex = srvDescHeap->GetNextAvailableIndex();
    D3D12_CPU_DESCRIPTOR_HANDLE objectSceneSRVCPUPos = srvDescHeap->GetCPUHandleAt(nextSrvIndex);
    objectSceneSRVCPUPos.ptr += dxContext->getDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    dxContext->getDevice()->CreateShaderResourceView(objectScenePositionTexture.Get(), &srvDesc, objectSceneSRVCPUPos);
    objectSceneSRVHandlePosition = srvDescHeap->GetGPUHandleAt(nextSrvIndex);
    objectSceneSRVHandlePosition.ptr += dxContext->getDevice()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    return true;
}

