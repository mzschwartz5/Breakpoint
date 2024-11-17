#include "ComputePipeline.h"

ComputePipeline::ComputePipeline(std::string rootSignatureShaderName, const std::string shaderFilePath, DXContext& context,
	CommandListID cmdID, D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: Pipeline(rootSignatureShaderName, context, cmdID, type, numberOfDescriptors, flags),
	computeShader(shaderFilePath)
{
	createPSOD();
	createPipelineState(context.getDevice());
}

void ComputePipeline::createPSOD()
{
	psoDesc.pRootSignature = rootSignature.Get();
	psoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader.getBuffer(), computeShader.getSize());

}

void ComputePipeline::createPipelineState(ComPointer<ID3D12Device6> device)
{
	device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
}