#include "DebugLayer.h"

bool DebugLayer::Init()
{
#ifdef _DEBUG
    // Init D3D12 Debug layer
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&m_d3d12Debug))))
    {
        m_d3d12Debug->EnableDebugLayer();

        // Init DXGI Debug
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&m_dxgiDebug))))
        {
            m_dxgiDebug->EnableLeakTrackingForThread();
            return true;
        }
    }
#endif

    return false;
}

void DebugLayer::Shutdown()
{
#ifdef _DEBUG
    if (m_dxgiDebug)
    {
        OutputDebugStringW(L"DXGI Reports living device objects:\n");
        /*m_dxgiDebug->ReportLiveObjects(
            DXGI_DEBUG_ALL, DXGI_DEBUG_RLO_FLAGS(DXGI_DEBUG_RLO_DETAIL | DXGI_DEBUG_RLO_IGNORE_INTERNAL)
        );*/
        OutputDebugStringW(L"this code is broken\n");
    }

    m_dxgiDebug.Release();
    m_d3d12Debug.Release();
#endif
}