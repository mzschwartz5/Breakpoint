#include "RenderPipeline.h"

RenderPipeline::RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context,
							   D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: vertexShader(vertexShaderName),
	  fragShader(fragShaderName),
	  rootSignatureShader(rootSignatureShaderName),
	  srvHeap(context, type, numberOfDescriptors, flags)
{
	context.getDevice()->CreateRootSignature(0, rootSignatureShader.getBuffer(), rootSignatureShader.getSize(), IID_PPV_ARGS(&rootSignature));
}

ComPointer<ID3D12RootSignature>& RenderPipeline::getRootSignature()
{
	return this->rootSignature;
}

ComPointer<ID3D12DescriptorHeap>& RenderPipeline::getSrvHeap()
{
	return srvHeap.Get();
}

void RenderPipeline::releaseResources()
{
	rootSignature.Release();
	srvHeap.releaseResources();
}