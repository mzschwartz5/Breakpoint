#pragma once

#include "../Support/WinInclude.h"
#include "../Support/Shader.h"
#include "../Support/ComPointer.h"
#include "../Support/Window.h"

class RenderPipeline {
public:
	RenderPipeline() = delete;
	RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context);

	ComPointer<ID3D12RootSignature>& getRootSignature();

	Shader& getVertexShader() { return vertexShader; }
	Shader& getFragmentShader() { return fragShader; }
private:
	Shader vertexShader, fragShader, rootSignatureShader;

	ComPointer<ID3D12RootSignature> rootSignature;
};