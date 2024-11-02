#include "DXContext.h"

DXContext::DXContext() {
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
    {
        //handle could not create device
    }

    D3D12_COMMAND_QUEUE_DESC cmdQueueDesc{};
    cmdQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    cmdQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    cmdQueueDesc.NodeMask = 0;
    cmdQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(device->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(&cmdQueue))))
    {
        //handle could not create command queue
    }

    if (FAILED(device->CreateFence(fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
    {
        //handle could not create fence
    }

    fenceEvent = CreateEvent(nullptr, false, false, nullptr);
    if (!fenceEvent)
    {
        //handle could not create fence event
    }

    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator))))
    {
        //handle cannot create cmd allocator
    }

    if (FAILED(device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmdList))))
    {
        //handle could n ot create cmd list
    }
}

DXContext::~DXContext() {
    cmdList.Release();
    cmdAllocator.Release();
    if (fenceEvent)
    {
        CloseHandle(fenceEvent);
    }
    cmdQueue.Release();
    device.Release();
}

void DXContext::signalAndWait() {
    cmdQueue->Signal(fence, ++fenceValue);
    if (SUCCEEDED(fence->SetEventOnCompletion(fenceValue, fenceEvent))) {
        if (WaitForSingleObject(fenceEvent, 20000) != WAIT_OBJECT_0) {
            std::exit(-1);
        }
    } else {
        std::exit(-1);
    }
}

ID3D12GraphicsCommandList5* DXContext::initCommandList()
{
    cmdAllocator->Reset();
    cmdList->Reset(cmdAllocator, nullptr);
    return cmdList;
}

void DXContext::executeCommandList() {
    if (SUCCEEDED(cmdList->Close())) {
        ID3D12CommandList* lists[] = { cmdList };
        cmdQueue->ExecuteCommandLists(1, lists);
        signalAndWait();
    }
}

ComPointer<ID3D12Device6>& DXContext::getDevice() {
    return device;
}

ComPointer<ID3D12CommandQueue>& DXContext::getCommandQueue() {
    return cmdQueue;
}
