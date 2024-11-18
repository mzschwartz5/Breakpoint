#include "IndexBuffer.h"

IndexBuffer::IndexBuffer(std::vector<unsigned int> indexData, const UINT indexDataSize)
    : indexData(indexData), indexDataSize(indexDataSize), uploadBuffer(), indexBuffer()
{}

D3D12_INDEX_BUFFER_VIEW IndexBuffer::passIndexDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList) {
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
    rd.Width = indexDataSize;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_NONE;

	if (FAILED(context.getDevice()->CreateCommittedResource(&hpUpload, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer)))) {
		throw std::runtime_error("Could not create committed resource for index buffer upload buffer");
	}
    
    if (FAILED(context.getDevice()->CreateCommittedResource(&hpDefault, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&indexBuffer)))) {
		throw std::runtime_error("Could not create committed resource for index buffer");
    }

    // Copy void* --> CPU Resource
    void* uploadBufferAddress;
    D3D12_RANGE uploadRange;
    uploadRange.Begin = 0;
    uploadRange.End = indexDataSize - 1;
    
    if (FAILED(uploadBuffer->Map(0, &uploadRange, &uploadBufferAddress))) {
		throw std::runtime_error("Could not map upload buffer");
    }
    
    memcpy(uploadBufferAddress, indexData.data(), indexDataSize);
    uploadBuffer->Unmap(0, &uploadRange);
    // Copy CPU Resource --> GPU Resource
    cmdList->CopyBufferRegion(indexBuffer, 0, uploadBuffer, 0, indexDataSize);

    // === Index buffer view ===
    D3D12_INDEX_BUFFER_VIEW ibv{};
    ibv.BufferLocation = indexBuffer->GetGPUVirtualAddress();
    ibv.SizeInBytes = indexDataSize;
    ibv.Format = DXGI_FORMAT_R32_UINT;
    return ibv;
}

ComPointer<ID3D12Resource1>& IndexBuffer::getUploadBuffer() {
    return uploadBuffer;
}

ComPointer<ID3D12Resource1>& IndexBuffer::getIndexBuffer() {
    return indexBuffer;
}

void IndexBuffer::releaseResources() {
    uploadBuffer.Release();
    indexBuffer.Release();
}
