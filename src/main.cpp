#include <iostream>
#include <d3d12.h>
#include <Windows.h>
#include <dxgi1_4.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

int main() {
    // Initialize the COM library
    CoInitialize(nullptr);

    // Create a Direct3D 12 device
    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(
        nullptr, // Default adapter
        D3D_FEATURE_LEVEL_12_0,
        IID_PPV_ARGS(&device)
    );

    // Check for successful device creation
    if (SUCCEEDED(hr)) {
        // D3D12 device created successfully
        std::cout << "hi";
    }

    // Clean up
    CoUninitialize();
    return 0;
}