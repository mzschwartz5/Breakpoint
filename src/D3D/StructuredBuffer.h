#pragma once

#include <vector>
#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "D3D/DXContext.h"
#include "D3D/Pipeline/Pipeline.h"
#include "DirectXMath.h"

using namespace DirectX;

// Create Enum for CBV, SRV, UAV
enum class BufferType {
	CBV,
	SRV,
	UAV
};

class StructuredBuffer {
public:
	StructuredBuffer() = default;
	StructuredBuffer(const void* data, unsigned int numEle, UINT eleSize, DescriptorHeap* heap);

	void copyDataFromGPU(DXContext& context, void* outputData, ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES state, CommandListID cmdId);

	ComPointer<ID3D12Resource1>& getBuffer();

	CD3DX12_GPU_DESCRIPTOR_HANDLE getGPUDescriptorHandle();

	D3D12_GPU_VIRTUAL_ADDRESS getGPUVirtualAddress();

	unsigned int getNumElements() { return numElements; }

	size_t getElementSize() { return elementSize; }


	void passCBVDataToGPU(DXContext& context);
	void passSRVDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList, CommandListID id);
	void passUAVDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList, CommandListID id);

	void releaseResources();

private:
	void findFreeHandle();

private:
	ComPointer<ID3D12Resource1> buffer;
	CD3DX12_CPU_DESCRIPTOR_HANDLE cpuHandle;
	CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle;
	DescriptorHeap* descriptorHeap = nullptr;
	const void* data;
	unsigned int numElements;
	UINT elementSize;
};