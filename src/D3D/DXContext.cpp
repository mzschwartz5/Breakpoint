#include "DXContext.h"

DXContext::DXContext() {

    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgiFactory)))) {
        //handle could not create dxgi factory
        throw std::runtime_error("Could not create dxgi factory");
    }

    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        //handle could not create device
        throw std::runtime_error("Could not create device");
    }

    D3D12_COMMAND_QUEUE_DESC cmdQueueDesc{};
    cmdQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    cmdQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    cmdQueueDesc.NodeMask = 0;
    cmdQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    if (FAILED(device->CreateCommandQueue(&cmdQueueDesc, IID_PPV_ARGS(&cmdQueue)))) {
        //handle could not create command queue
        throw std::runtime_error("Could not create command queue");
    }

    if (FAILED(device->CreateFence(fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        //handle could not create fence
        throw std::runtime_error("Could not create fence");
    }

    fenceEvent = CreateEvent(nullptr, false, false, nullptr);
    if (!fenceEvent) {
        //handle could not create fence event
        throw std::runtime_error("Could not create fence event");
    }

	for (int i = 0; i < NUM_CMDLISTS; i++) {
		ComPointer<ID3D12CommandAllocator> cmdAllocator;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAllocator)))) {
			//handle cannot create cmd allocator
			throw std::runtime_error("Could not create command allocator");
		}
		cmdAllocators[i] = cmdAllocator;
	}

}

DXContext::~DXContext() {
    for (auto& cmdList : cmdLists) {
        cmdList.Release();
    }

    for (auto& cmdAllocator : cmdAllocators) {
        cmdAllocator.Release();
    }

    if (fenceEvent)
    {
        CloseHandle(fenceEvent);
    }
    fence.Release();
    cmdQueue.Release();
    device.Release();
    dxgiFactory.Release();
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

void DXContext::resetCommandList(CommandListID id)
{
    cmdAllocators[id]->Reset();
    
	cmdLists[id]->Reset(cmdAllocators[id], nullptr);
}

void DXContext::executeCommandLists() {
    for (auto& cmdList : cmdLists) {
        if (SUCCEEDED(cmdList->Close())) {
            ID3D12CommandList* lists[] = { cmdList };
            cmdQueue->ExecuteCommandLists(1, lists);
            signalAndWait();
        }
    }
}

void DXContext::executeCommandList(CommandListID id) {
	if (SUCCEEDED(cmdLists[id]->Close())) {
		ID3D12CommandList* lists[] = { cmdLists[id] };
		cmdQueue->ExecuteCommandLists(1, lists);
		signalAndWait();
	}
}

void DXContext::flush(size_t count) {
    for (size_t i = 0; i < count; i++) {
        signalAndWait();
    }
}

void DXContext::signalAndWaitForFence(ComPointer<ID3D12Fence>& fence, UINT64& fenceValue) {
	cmdQueue->Signal(fence, fenceValue);
    if (fence->GetCompletedValue() < fenceValue) {
        HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (eventHandle == nullptr) {
            throw std::runtime_error("Failed to create event handle.");
        }

        // Set the event to be triggered when the GPU reaches the fence value
        fence->SetEventOnCompletion(fenceValue, eventHandle);

        // Wait until the event is triggered, meaning the GPU has finished
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }
    fenceValue++;
}

ComPointer<IDXGIFactory7>& DXContext::getFactory() {
    return dxgiFactory;
}

ComPointer<ID3D12Device6>& DXContext::getDevice() {
    return device;
}

ComPointer<ID3D12CommandQueue>& DXContext::getCommandQueue() {
    return cmdQueue;
}

ID3D12GraphicsCommandList5* DXContext::createCommandList(CommandListID id)
{
    ComPointer<ID3D12GraphicsCommandList5> cmdList;
    if (FAILED(device->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&cmdList)))) {
        //handle could not create cmd list
        throw std::runtime_error("Could not create command list");
    }

    cmdLists[id] = cmdList;
    return cmdList.Get();
}
