#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"
#include "D3D/RenderPipeline.h"
#include "DirectXMath.h"

using namespace DirectX;

class ModelMatrixBuffer {
public:
	ModelMatrixBuffer() = delete;
	ModelMatrixBuffer(std::vector<XMFLOAT4X4> &matrices, size_t instanceSize);

	void passModelMatrixDataToGPU(DXContext& context, RenderPipeline& pipeline, ID3D12GraphicsCommandList5* cmdList);

	ComPointer<ID3D12Resource1>& getModelMatrixBuffer();

	void releaseResources();

private:
	ComPointer<ID3D12Resource1> modelMatrixBuffer;

	std::vector<XMFLOAT4X4>* modelMatrices;

	size_t instanceCount;
};