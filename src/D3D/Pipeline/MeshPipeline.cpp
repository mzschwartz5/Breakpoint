#include "MeshPipeline.h"

MeshPipeline::MeshPipeline(std::string meshShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context,
    CommandListID cmdID, D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: Pipeline(rootSignatureShaderName, context, cmdID, type, numberOfDescriptors, flags), 
      meshShader(meshShaderName),
      fragShader(fragShaderName),
      samplerHeap(context, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
{
    // TODO: this should be in the base pipeline class (same for compute pipeline)
    createPSOD();
    createPipelineState(context.getDevice());
    createSampler(context.getDevice());
}

void MeshPipeline::createPSOD() {
    psod.pRootSignature = rootSignature;
    psod.MS.BytecodeLength = meshShader.getSize();
    psod.MS.pShaderBytecode = meshShader.getBuffer();
    psod.PS.BytecodeLength = fragShader.getSize();
    psod.PS.pShaderBytecode = fragShader.getBuffer();
    psod.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psod.NumRenderTargets = 1;
    psod.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psod.SampleDesc.Count = 1;
    psod.SampleMask = UINT_MAX;
    psod.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psod.RasterizerState.FrontCounterClockwise = TRUE;
    psod.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;    
    psod.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psod.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
}

void MeshPipeline::createPipelineState(ComPointer<ID3D12Device6> device) {
    CD3DX12_PIPELINE_MESH_STATE_STREAM psoStream = CD3DX12_PIPELINE_MESH_STATE_STREAM(psod);
    D3D12_PIPELINE_STATE_STREAM_DESC streamDesc = { sizeof(psoStream), &psoStream };
    HRESULT hr = device->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&pso));

    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create compute pipeline state");
    }
}

void MeshPipeline::createSampler(ComPointer<ID3D12Device6> device) {
    samplerHeap.Get()->SetName(L"Sampler Heap");
    unsigned int nextHeapIdx = samplerHeap.GetNextAvailableIndex();
    samplerHandle = samplerHeap.GetGPUHandleAt(nextHeapIdx);

    D3D12_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D12_FLOAT32_MAX;
    samplerDesc.MaxAnisotropy = 1;
    samplerDesc.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    device->CreateSampler(&samplerDesc, samplerHeap.GetCPUHandleAt(nextHeapIdx));
}
