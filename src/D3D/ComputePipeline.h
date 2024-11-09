#pragma once
#include <string>
#include "../Support/WinInclude.h"
#include "../Support/Shader.h"
#include "../Support/ComPointer.h"
#include "../Support/Window.h"
#include "DescriptorHeap.h"

class ComputePipeline
{
public:
	ComputePipeline(std::string rootSignatureShaderName, const std::string& shaderFilePath, DXContext& context,
		D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags);

	ID3D12PipelineState* GetAddress();
	ComPointer<ID3D12RootSignature>& getRootSignature();
	ComPointer<ID3D12DescriptorHeap>& getSrvHeap();

	Shader& getComputeShader() { return computeShader; }

	void releaseResources();

private:
	void CreatePipelineState(DXContext& context, ComPointer<ID3D12RootSignature> rootSignature);

private:
	Shader computeShader, rootSignatureShader;
	ComPointer<ID3D12RootSignature> rootSignature;
	ComPointer<ID3D12PipelineState> pipeline;
	DescriptorHeap srvHeap;
};