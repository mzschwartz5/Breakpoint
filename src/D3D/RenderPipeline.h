#pragma once

#include "../Support/WinInclude.h"
#include "../Support/Shader.h"
#include "../Support/ComPointer.h"
#include "../Support/Window.h"

//D3D12_INPUT_ELEMENT_DESC vertexLayout[] =
//{
//	{ "Position", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
//};

enum VertexLayoutType {
	Standard2D,
};

class RenderPipeline {
public:
	RenderPipeline() = delete;
	RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, VertexLayoutType layoutType, DXContext& context);

	D3D12_GRAPHICS_PIPELINE_STATE_DESC& getPSOD();

	ComPointer<ID3D12RootSignature>& getRootSignature();

	Shader& getVertexShader() { return vertexShader; }
	Shader& getFragShader() { return fragShader; }
private:
	Shader vertexShader, fragShader, rootSignatureShader;

	VertexLayoutType layoutType;

	ComPointer<ID3D12RootSignature> rootSignature;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxPsod;
};