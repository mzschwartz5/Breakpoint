#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"
#include "DirectXMath.h"

using namespace DirectX;

class VertexBuffer {
public:
	VertexBuffer() = default;
	VertexBuffer(std::vector<XMFLOAT3> vertexData, size_t vertexDataSize, size_t vertexSize);

	D3D12_VERTEX_BUFFER_VIEW passVertexDataToGPU(DXContext& context, ID3D12GraphicsCommandList5* cmdList);

	ComPointer<ID3D12Resource1>& getUploadBuffer();
	ComPointer<ID3D12Resource1>& getVertexBuffer();

	void releaseResources();

private:
	ComPointer<ID3D12Resource1> uploadBuffer;
	ComPointer<ID3D12Resource1> vertexBuffer;

	size_t vertexDataSize;
	size_t vertexSize;
	std::vector<XMFLOAT3> vertexData;
};