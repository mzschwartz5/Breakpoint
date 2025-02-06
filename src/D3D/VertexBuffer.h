#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"
#include "DirectXMath.h"

using namespace DirectX;
struct Vertex {
	XMFLOAT4 pos;
	XMFLOAT4 nor;
};

class VertexBuffer {
public:
	VertexBuffer() = default;
	VertexBuffer(std::vector<Vertex> vertexData, UINT vertexDataSize, UINT vertexSize);

	D3D12_VERTEX_BUFFER_VIEW passVertexDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList);

	ComPointer<ID3D12Resource1>& getUploadBuffer();
	ComPointer<ID3D12Resource1>& getVertexBuffer();

	void releaseResources();

private:
	ComPointer<ID3D12Resource1> uploadBuffer;
	ComPointer<ID3D12Resource1> vertexBuffer;

	UINT vertexDataSize;
	UINT vertexSize;
	std::vector<Vertex> vertexData;
};