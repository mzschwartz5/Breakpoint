#pragma once

#include "../Support/WinInclude.h"
#include "../Support/Shader.h"
#include "../Support/ComPointer.h"
#include "../Support/Window.h"

enum VertexLayoutType {
	Standard2D,
};

class RenderPipeline {
public:
	RenderPipeline() = delete;
	RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, VertexLayoutType layoutType, DXContext& context);

	ComPointer<ID3D12RootSignature>& getRootSignature();

	Shader& getVertexShader() { return vertexShader; }
	Shader& getFragmentShader() { return fragShader; }
private:
	Shader vertexShader, fragShader, rootSignatureShader;

	VertexLayoutType layoutType;

	ComPointer<ID3D12RootSignature> rootSignature;
};