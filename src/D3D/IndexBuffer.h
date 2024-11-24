#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"

class IndexBuffer {
public:
	IndexBuffer() = default;
	IndexBuffer(std::vector<unsigned int> indexData, const UINT indexDataSize);

	D3D12_INDEX_BUFFER_VIEW passIndexDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList);

	ComPointer<ID3D12Resource1>& getUploadBuffer();
	ComPointer<ID3D12Resource1>& getIndexBuffer();

	void releaseResources();

private:
	ComPointer<ID3D12Resource1> uploadBuffer;
	ComPointer<ID3D12Resource1> indexBuffer;

	UINT indexDataSize;
	std::vector<unsigned int> indexData;
};