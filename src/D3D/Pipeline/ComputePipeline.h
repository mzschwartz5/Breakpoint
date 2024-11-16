#pragma once
#include <string>
#include "Pipeline.h"
#include "../../Support/WinInclude.h"
#include "../../Support/Shader.h"
#include "../../Support/ComPointer.h"
#include "../../Support/Window.h"
#include "../DescriptorHeap.h"

class ComputePipeline : public Pipeline
{
public: 
	ComputePipeline() = delete;
	ComputePipeline(std::string rootSignatureShaderName, const std::string shaderFilePath, DXContext& context, CommandListID cmdID,
		D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags);

	Shader& getComputeShader() { return computeShader; }
	void createPSOD() override;
	void createPipelineState(ComPointer<ID3D12Device6> device) override;

private:
	Shader computeShader;
	D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
};