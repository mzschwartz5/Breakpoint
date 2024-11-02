#pragma once
#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include <stdexcept>

class DXContext
{
public:
    DXContext();
    ~DXContext();

    void signalAndWait();
    ID3D12GraphicsCommandList5* initCommandList();
    void executeCommandList();

    ComPointer<ID3D12Device6>& getDevice();
    ComPointer<ID3D12CommandQueue>& getCommandQueue();

private:
    ComPointer<ID3D12Device6> device;
    ComPointer<ID3D12CommandQueue> cmdQueue;
    ComPointer<ID3D12CommandAllocator> cmdAllocator;
    ComPointer<ID3D12GraphicsCommandList5> cmdList;
    ComPointer<ID3D12Fence1> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

};