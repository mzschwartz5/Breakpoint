#include "DebugLayer.h"

DebugLayer::DebugLayer() {
#ifdef _DEBUG
    // Init D3D12 Debug layer
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3d12Debug))))
    {
        d3d12Debug->EnableDebugLayer();

        // Init DXGI Debug
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgiDebug))))
        {
            dxgiDebug->EnableLeakTrackingForThread();
            initialized = true;
        }
    }
#endif

    initialized = false;
}

DebugLayer::~DebugLayer() {
#ifdef _DEBUG
    if (dxgiDebug)
    {
        OutputDebugStringW(L"DXGI Reports living device objects:\n");
        dxgiDebug->ReportLiveObjects(
            DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_DETAIL | DXGI_DEBUG_RLO_IGNORE_INTERNAL)
        );
    }

    dxgiDebug.Release();
    d3d12Debug.Release();
#endif
}

bool DebugLayer::isInitialized() {
    return initialized;
}
