#include "StructuredBuffer.h"

StructuredBuffer::StructuredBuffer(const void* inputData, unsigned int numEle, size_t eleSize)
	: data(inputData), numElements(numEle), elementSize(eleSize)
{}

D3D12_GPU_VIRTUAL_ADDRESS StructuredBuffer::passCBVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle) {
    // Calculate the aligned buffer size (256-byte alignment required for CBV)
    UINT bufferSize = (numElements * elementSize + 255) & ~255; // Round up to 256 bytes

    // Step 1: Create a committed resource for the CBV (Upload heap for CPU access)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,  // State for CPU-read access
        nullptr,
        IID_PPV_ARGS(&buffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create CBV buffer.");
    }

    // Step 2: Map the buffer and copy data directly into it (no upload buffer needed for CBV)
    void* mappedData = nullptr;
    buffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, data, numElements * elementSize); // Copy only actual data size
    buffer->Unmap(0, nullptr);

    // Step 3: Create the CBV in the descriptor heap
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
    cbvDesc.BufferLocation = buffer->GetGPUVirtualAddress();
    cbvDesc.SizeInBytes = bufferSize; // Must be 256-byte aligned

    context.getDevice()->CreateConstantBufferView(&cbvDesc, descriptorHandle);

	return buffer->GetGPUVirtualAddress();
}

D3D12_GPU_VIRTUAL_ADDRESS StructuredBuffer::passSRVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle, ID3D12GraphicsCommandList5* cmdList) {
	// THIS FUNCTION WILL RESET THE COMMAND LIST AT THE END OF THE CALL

    // Calculate the total buffer size
    UINT bufferSize = numElements * elementSize;

    // Step 1: Create a default heap resource for the SRV (GPU-only)
    D3D12_HEAP_PROPERTIES defaultHeapProps = {};
    defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;  // No unordered access flag

    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&buffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create SRV buffer.");
    }

    // Step 2: Create an upload buffer to copy `data` to the GPU buffer
    D3D12_HEAP_PROPERTIES uploadHeapProps = {};
    uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC uploadDesc = bufferDesc;
    uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE; // Upload buffer has no flags

    ComPointer<ID3D12Resource> uploadBuffer;
    hr = context.getDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create SRV upload buffer.");
    }

    // Step 3: Copy data into the upload buffer
    void* mappedData = nullptr;
    uploadBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, data, bufferSize); // Assume `data` is correctly sized
    uploadBuffer->Unmap(0, nullptr);

    // Step 4: Copy data from the upload buffer to the GPU buffer
    cmdList->CopyResource(buffer.Get(), uploadBuffer.Get());

    // Step 5: Transition the SRV buffer to the PIXEL_SHADER_RESOURCE state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // Create a fence to wait for the GPU to finish copying data
    ComPointer<ID3D12Fence> fence;
    UINT64 fenceValue = 1;
    hr = context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create fence.");
    }

    context.executeCommandList();
    context.getCommandQueue()->Signal(fence.Get(), fenceValue);

    // Wait for the fence to reach the signaled value
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

    // Step 6: Create the SRV in the descriptor heap
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = numElements;
    srvDesc.Buffer.StructureByteStride = elementSize;

    context.getDevice()->CreateShaderResourceView(buffer.Get(), &srvDesc, descriptorHandle);

    cmdList = context.initCommandList();

	return buffer->GetGPUVirtualAddress();
}

D3D12_GPU_VIRTUAL_ADDRESS StructuredBuffer::passUAVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle, ID3D12GraphicsCommandList5 *cmdList) {
    // THIS FUNCTION WILL RESET THE COMMAND LIST AT THE END OF THE CALL

    // Calculate the total buffer size
    UINT bufferSize = numElements * elementSize;

    // Create a default heap resource for the UAV (GPU-only)
    D3D12_HEAP_PROPERTIES defaultHeapProps = {};
    defaultHeapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = bufferSize;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &defaultHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&buffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create UAV buffer.");
    }

    // Create an upload buffer to copy `data` to the GPU buffer
    D3D12_HEAP_PROPERTIES uploadHeapProps = {};
    uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC uploadDesc = {};
    uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadDesc.Width = bufferSize;
    uploadDesc.Height = 1;
    uploadDesc.DepthOrArraySize = 1;
    uploadDesc.MipLevels = 1;
    uploadDesc.SampleDesc.Count = 1;
    uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	uploadDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPointer<ID3D12Resource> uploadBuffer;
    hr = context.getDevice()->CreateCommittedResource(
        &uploadHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &uploadDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBuffer)
    );
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create UAV upload buffer.");
    }

    // Copy data into the upload buffer
    void* mappedData = nullptr;
    uploadBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, data, bufferSize); // Assume `data` is correctly sized
    uploadBuffer->Unmap(0, nullptr);

    // Copy data from the upload buffer to the GPU buffer
    cmdList->CopyResource(buffer.Get(), uploadBuffer.Get());

	// Transition the UAV buffer to the unordered access state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

	// Create a fence to wait for the GPU to finish copying data
    ComPointer<ID3D12Fence> fence;
    UINT64 fenceValue = 1;
    hr = context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create fence.");
    }

    context.executeCommandList();
    context.getCommandQueue()->Signal(fence.Get(), fenceValue);

    // Wait for the fence to reach the signaled value
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

    // Create the UAV in the descriptor heap
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = numElements;
    uavDesc.Buffer.StructureByteStride = elementSize;

    context.getDevice()->CreateUnorderedAccessView(buffer.Get(), nullptr, &uavDesc, descriptorHandle);

	// Reset the command list
    cmdList = context.initCommandList();

    return buffer->GetGPUVirtualAddress();
}

void StructuredBuffer::copyDataFromGPU(DXContext& context, void* outputData, ID3D12GraphicsCommandList5* cmdList, D3D12_RESOURCE_STATES state) {
	// THIS FUNCTION WILL RESET THE COMMAND LIST AT THE END OF THE CALL

    // Create a readback buffer to copy data from the GPU buffer
    ComPointer<ID3D12Resource> readbackBuffer;

    // Set up heap properties for a readback buffer
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;  // CPU-accessible for readback

    // Describe the buffer resource
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = elementSize * numElements;  // Match the size of the GPU buffer
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Create the readback buffer resource
    HRESULT hr = context.getDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,  // Must be in COPY_DEST state to receive copied data
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)
    );

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create readback buffer.");
    }

    // 2. Transition the buffer to COPY_SOURCE state
    D3D12_RESOURCE_BARRIER uavToCopySourceBarrier = {};
    uavToCopySourceBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    uavToCopySourceBarrier.Transition.pResource = buffer.Get();
    uavToCopySourceBarrier.Transition.StateBefore = state;
    uavToCopySourceBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    uavToCopySourceBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &uavToCopySourceBarrier);

    // Copy the data from the GPU buffer to the readback buffer
    cmdList->CopyResource(readbackBuffer.Get(), buffer.Get());

	// Transition the buffer back to its original state
    D3D12_RESOURCE_BARRIER copySourceToUavBarrier = {};
    copySourceToUavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    copySourceToUavBarrier.Transition.pResource = buffer.Get();
    copySourceToUavBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    copySourceToUavBarrier.Transition.StateAfter = state;
    copySourceToUavBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &copySourceToUavBarrier);

    // Create a fence
    UINT64 fenceValue = 1;
    ComPointer<ID3D12Fence> fence;
    context.getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Execute the command list to perform the copy operation
    context.executeCommandList();
    context.getCommandQueue()->Signal(fence.Get(), fenceValue);
    context.flush(1);

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

    // Map the readback buffer to access the data on the CPU
    void* mappedData = nullptr;
    D3D12_RANGE readRange{ 0, elementSize * numElements }; // The range of the buffer to map
    hr = readbackBuffer->Map(0, &readRange, &mappedData);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to map readback buffer.");
    }
    // Copy data from the mapped readback buffer to outputData
    memcpy(outputData, mappedData, elementSize * numElements);

    // Unmap the readback buffer
    D3D12_RANGE writeRange{ 0, 0 }; // Indicate no data written by CPU
    readbackBuffer->Unmap(0, &writeRange);

    cmdList = context.initCommandList();
}

ComPointer<ID3D12Resource1>& StructuredBuffer::getBuffer()
{
	return this->buffer;
}

void StructuredBuffer::releaseResources()
{
	this->buffer.Release();
}