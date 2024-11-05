#include "RenderPipeline.h"

RenderPipeline::RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, VertexLayoutType layoutType, DXContext& context) 
	: vertexShader(vertexShaderName),
	  fragShader(fragShaderName),
	  rootSignatureShader(rootSignatureShaderName),
	  layoutType(layoutType)
{
	context.getDevice()->CreateRootSignature(0, rootSignatureShader.getBuffer(), rootSignatureShader.getSize(), IID_PPV_ARGS(&rootSignature));
}

ComPointer<ID3D12RootSignature>& RenderPipeline::getRootSignature()
{
	return this->rootSignature;
}