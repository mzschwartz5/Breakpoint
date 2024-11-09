#include "ComputePipeline.h"

ComputePipeline::ComputePipeline(std::string rootSignatureShaderName, const std::string shaderFilePath, DXContext& context,
	D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: computeShader(shaderFilePath),
	rootSignatureShader(rootSignatureShaderName),
	descriptorHeap(context, type, numberOfDescriptors, flags)
{
	context.getDevice()->CreateRootSignature(0, rootSignatureShader.getBuffer(), rootSignatureShader.getSize(), IID_PPV_ARGS(&rootSignature));
	CreatePipelineState(context);
}

ID3D12PipelineState* ComputePipeline::GetAddress()
{
	return pipeline.Get();
}

ComPointer<ID3D12RootSignature> ComputePipeline::getRootSignature()
{
	// Return a pointer to the root signature
	return rootSignature;
}

ComPointer<ID3D12DescriptorHeap> ComputePipeline::getDescriptorHeap()
{
	return descriptorHeap.Get();
}

void ComputePipeline::releaseResources()
{
	rootSignature.Release();
	pipeline.Release();
	descriptorHeap.releaseResources();
}

void ComputePipeline::CreatePipelineState(DXContext& context)
{
	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
	psoDesc.pRootSignature = rootSignature.Get();
	psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.getBuffer(), computeShader.getSize());

	context.getDevice()->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline));
}