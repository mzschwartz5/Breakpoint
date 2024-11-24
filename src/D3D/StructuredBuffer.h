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

	StructuredBuffer(const void* data, unsigned int numEle, UINT eleSize);

	void copyDataFromGPU(DXContext& context, void* outputData, ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES state, CommandListID cmdId);

	ComPointer<ID3D12Resource1>& getBuffer();

	CD3DX12_CPU_DESCRIPTOR_HANDLE getUAVCPUDescriptorHandle();
	CD3DX12_CPU_DESCRIPTOR_HANDLE getSRVCPUDescriptorHandle();
	CD3DX12_CPU_DESCRIPTOR_HANDLE getCBVCPUDescriptorHandle();

	CD3DX12_GPU_DESCRIPTOR_HANDLE getUAVGPUDescriptorHandle();
	CD3DX12_GPU_DESCRIPTOR_HANDLE getSRVGPUDescriptorHandle();
	CD3DX12_GPU_DESCRIPTOR_HANDLE getCBVGPUDescriptorHandle();

	D3D12_GPU_VIRTUAL_ADDRESS getGPUVirtualAddress();

	unsigned int getNumElements() { return numElements; }

	size_t getElementSize() { return elementSize; }

	void passCBVDataToGPU(DXContext& context, DescriptorHeap* dh);
	void passDataToGPU(DXContext& context, ID3D12GraphicsCommandList6* cmdList, CommandListID id);
	void createUAV(DXContext& context, DescriptorHeap* dh);
	void createSRV(DXContext& context, DescriptorHeap* dh);

	void releaseResources();

private:
	void findFreeHandle(DescriptorHeap* dh, CD3DX12_CPU_DESCRIPTOR_HANDLE& cpuHandle, CD3DX12_GPU_DESCRIPTOR_HANDLE& gpuHandle);

private:
	ComPointer<ID3D12Resource1> buffer;

	CD3DX12_CPU_DESCRIPTOR_HANDLE UAVcpuHandle;
	CD3DX12_CPU_DESCRIPTOR_HANDLE SRVcpuHandle;
	CD3DX12_CPU_DESCRIPTOR_HANDLE CBVcpuHandle;

	CD3DX12_GPU_DESCRIPTOR_HANDLE UAVgpuHandle;
	CD3DX12_GPU_DESCRIPTOR_HANDLE SRVgpuHandle;
	CD3DX12_GPU_DESCRIPTOR_HANDLE CBVgpuHandle;

	bool isCBV = false;
	bool isUAV = false;
	bool isSRV = false;

	const void* data;
	unsigned int numElements;
	UINT elementSize;
};