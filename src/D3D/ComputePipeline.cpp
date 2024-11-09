#include "ComputePipeline.h"

ComputePipeline::ComputePipeline(std::string rootSignatureShaderName, const std::string& shaderFilePath, DXContext& context,
	D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: computeShader(shaderFilePath),
	rootSignatureShader(rootSignatureShaderName),
	srvHeap(context, type, numberOfDescriptors, flags)
{
	context.getDevice()->CreateRootSignature(0, rootSignatureShader.getBuffer(), rootSignatureShader.getSize(), IID_PPV_ARGS(&rootSignature));
	CreatePipelineState(context, rootSignature);
}

ID3D12PipelineState* ComputePipeline::GetAddress()
{
	return pipeline.Get();
}

ComPointer<ID3D12RootSignature>& ComputePipeline::getRootSignature()
{
	return rootSignature;
}

ComPointer<ID3D12DescriptorHeap>& ComputePipeline::getSrvHeap()
{
	return srvHeap.Get();
}

void ComputePipeline::releaseResources()
{
	rootSignature.Release();
	pipeline.Release();
	srvHeap.releaseResources();
}

void ComputePipeline::CreatePipelineState(DXContext& context, ComPointer<ID3D12RootSignature> rootSignature)
{
	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = rootSignature.Get();
	psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.getBuffer(), computeShader.getSize());

	context.getDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline));
}