#pragma once
#include "Pipeline.h"

class MeshPipeline : public Pipeline {
public:
	MeshPipeline() = delete;
	MeshPipeline(std::string meshShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context,
		CommandListID cmdID, D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags);

	Shader& getMeshShader() { return meshShader; }

	void createPSOD() override;

	void createPipelineState(ComPointer<ID3D12Device6> device) override;


	D3D12_GPU_DESCRIPTOR_HANDLE getSamplerHandle() { return samplerHandle; }
	DescriptorHeap* getSamplerHeap() { return &samplerHeap; }

private: 
	Shader meshShader, fragShader;

	DescriptorHeap samplerHeap;
	D3D12_GPU_DESCRIPTOR_HANDLE samplerHandle;

	D3DX12_MESH_SHADER_PIPELINE_STATE_DESC psod{};
	void createSampler(ComPointer<ID3D12Device6> device);
};