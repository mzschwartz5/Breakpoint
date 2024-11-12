#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"
#include "D3D/Pipeline/Pipeline.h"
#include "DirectXMath.h"

using namespace DirectX;

class StructuredBuffer {
public:
	StructuredBuffer() = default;
	StructuredBuffer(const void* data, unsigned int numEle, size_t eleSize);

	D3D12_GPU_VIRTUAL_ADDRESS passCBVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle);
	D3D12_GPU_VIRTUAL_ADDRESS passSRVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle, ID3D12GraphicsCommandList5 *cmdList, CommandListID id);
	D3D12_GPU_VIRTUAL_ADDRESS passUAVDataToGPU(DXContext& context, D3D12_CPU_DESCRIPTOR_HANDLE descriptorHandle, ID3D12GraphicsCommandList5 *cmdList, CommandListID id);
	void copyDataFromGPU(DXContext& context, void* outputData, ID3D12GraphicsCommandList5* cmdList, D3D12_RESOURCE_STATES state);

	ComPointer<ID3D12Resource1>& getBuffer();

	unsigned int getNumElements() { return numElements; }

	size_t getElementSize() { return elementSize; }

	void releaseResources();

private:
	ComPointer<ID3D12Resource1> buffer;

	const void* data;
	unsigned int numElements;
	size_t elementSize;
};