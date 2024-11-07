#include "ModelMatrixBuffer.h"

ModelMatrixBuffer::ModelMatrixBuffer(std::vector<XMFLOAT4X4>& matrices, size_t instanceSize)
	: modelMatrices(&matrices), instanceCount(instanceSize)
{}

void ModelMatrixBuffer::passModelMatrixDataToGPU(DXContext& context, RenderPipeline& pipeline, ID3D12GraphicsCommandList5* cmdList)
{
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = sizeof(XMFLOAT4X4) * instanceCount;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    context.getDevice()->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&modelMatrixBuffer)
    );

    // Copy model matrices data to the buffer
    void* mappedData;
    modelMatrixBuffer->Map(0, nullptr, &mappedData);
    memcpy(mappedData, modelMatrices->data(), sizeof(XMFLOAT4X4) * instanceCount);
    modelMatrixBuffer->Unmap(0, nullptr);

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = static_cast<UINT>(instanceCount);
    srvDesc.Buffer.StructureByteStride = sizeof(XMFLOAT4X4);
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    context.getDevice()->CreateShaderResourceView(modelMatrixBuffer.Get(), &srvDesc, pipeline.getSrvHeap()->GetCPUDescriptorHandleForHeapStart());
}

ComPointer<ID3D12Resource1>& ModelMatrixBuffer::getModelMatrixBuffer()
{
	return this->modelMatrixBuffer;
}

void ModelMatrixBuffer::releaseResources()
{
	this->modelMatrixBuffer.Release();
}