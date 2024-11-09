#pragma once

#include "../Support/WinInclude.h"
#include "../Support/Shader.h"
#include "../Support/ComPointer.h"
#include "../Support/Window.h"
#include "DescriptorHeap.h"

class RenderPipeline {
public:
	RenderPipeline() = delete;
	RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context,
				   D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags);

	ComPointer<ID3D12RootSignature>& getRootSignature();
	ComPointer<ID3D12DescriptorHeap>& getDescriptorHeap();

	Shader& getVertexShader() { return vertexShader; }
	Shader& getFragmentShader() { return fragShader; }

	void releaseResources();

private:
	Shader vertexShader, fragShader, rootSignatureShader;

	ComPointer<ID3D12RootSignature> rootSignature;
	DescriptorHeap descriptorHeap;
};