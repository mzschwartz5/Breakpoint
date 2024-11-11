#pragma once
#include "../Support/WinInclude.h"
#include "../Support/ComPointer.h"
#include <stdexcept>
#include <array>

#define NUM_CMDLISTS 3
enum CommandListID {
    RENDER_ID,
    MESH_ID,
    PAPA_ID
};

class DXContext
{
public:
    DXContext();
    ~DXContext();

    void signalAndWait();
    void resetCommandLists();
    void executeCommandLists();

    void flush(size_t count);

    ComPointer<IDXGIFactory7>& getFactory();
    ComPointer<ID3D12Device6>& getDevice();
    ComPointer<ID3D12CommandQueue>& getCommandQueue();
    ID3D12GraphicsCommandList6* createCommandList(CommandListID id);

private:
    ComPointer<IDXGIFactory7> dxgiFactory;

    ComPointer<ID3D12Device6> device;

    ComPointer<ID3D12CommandQueue> cmdQueue;
    ComPointer<ID3D12CommandAllocator> cmdAllocator;
    std::array<ComPointer<ID3D12GraphicsCommandList6>, NUM_CMDLISTS> cmdLists{};

    ComPointer<ID3D12Fence1> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

};