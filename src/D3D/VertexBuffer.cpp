#include "VertexBuffer.h"

VertexBuffer::VertexBuffer(float* vertexData, size_t vertexDataSize, size_t vertexSize) 
    : vertexData(vertexData), vertexDataSize(vertexDataSize), vertexSize(vertexSize), uploadBuffer(), vertexBuffer()
{}

D3D12_VERTEX_BUFFER_VIEW VertexBuffer::passVertexDataToGPU(DXContext& context, ID3D12GraphicsCommandList5* cmdList) {
    D3D12_HEAP_PROPERTIES hpUpload{};
    hpUpload.Type = D3D12_HEAP_TYPE_UPLOAD;
    hpUpload.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hpUpload.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hpUpload.CreationNodeMask = 0;
    hpUpload.VisibleNodeMask = 0;
    D3D12_HEAP_PROPERTIES hpDefault{};
    hpDefault.Type = D3D12_HEAP_TYPE_DEFAULT;
    hpDefault.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hpDefault.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hpDefault.CreationNodeMask = 0;
    hpDefault.VisibleNodeMask = 0;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    rd.Width = vertexDataSize;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

    HRESULT hr;

    hr = context.getDevice()->CreateCommittedResource(&hpUpload, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer));
	if (FAILED(hr)) {
		throw std::runtime_error("Could not create committed resource for vertex buffer upload buffer");
	}
    
    hr = context.getDevice()->CreateCommittedResource(&hpDefault, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&vertexBuffer));
    if (FAILED(hr)) {
		throw std::runtime_error("Could not create committed resource for vertex buffer");
    }
    
    // Copy void* --> CPU Resource
    void* uploadBufferAddress;
    D3D12_RANGE uploadRange;
    uploadRange.Begin = 0;
    uploadRange.End = vertexDataSize - 1;
    hr = uploadBuffer->Map(0, &uploadRange, &uploadBufferAddress);
    if (FAILED(hr)) {
		throw std::runtime_error("Could not map upload buffer");
    }

    memcpy(uploadBufferAddress, vertexData, vertexDataSize);
    uploadBuffer->Unmap(0, &uploadRange);
    // Copy CPU Resource --> GPU Resource
    cmdList->CopyBufferRegion(vertexBuffer, 0, uploadBuffer, 0, vertexDataSize);
    
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = vertexBuffer.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    // === Vertex buffer view ===
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    vbv.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
    vbv.SizeInBytes = vertexDataSize;
    vbv.StrideInBytes = vertexSize;
    return vbv;
}

ComPointer<ID3D12Resource1>& VertexBuffer::getUploadBuffer() {
	return uploadBuffer;
}

ComPointer<ID3D12Resource1>& VertexBuffer::getVertexBuffer() {
	return vertexBuffer;
}

void VertexBuffer::releaseResources() {
    uploadBuffer.Release();
    vertexBuffer.Release();
}
