#include "PBMPMScene.h"

PBMPMScene::PBMPMScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* compPipeline, ID3D12GraphicsCommandList5* cmdList, unsigned int instances)
	: Scene(context, pipeline, cmdList), context(context), pipeline(pipeline), 
	computePipeline(compPipeline), cmdList(cmdList), instanceCount(instances),
	modelMat(XMMatrixIdentity())
{}

void PBMPMScene::constructScene() {
	// Create Model Matrix
	modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

	// Create position and velocity data
	for (int i = 0; i < instanceCount; ++i) {
		positions.push_back({ -0.72f + 0.15f * i, 0.f, 0.f });
	}

	for (int i = 0; i < instanceCount; ++i) {
		velocities.push_back({ 0.0f, 0.0f, 0.0f });
	}

	// Create Structured Buffers
	positionBuffer = StructuredBuffer(positions.data(), instanceCount, sizeof(XMFLOAT3));
	velocityBuffer = StructuredBuffer(velocities.data(), instanceCount, sizeof(XMFLOAT3));

	// Pass Structured Buffers to Compute Pipeline
	positionBuffer.passUAVDataToGPU(*context, computePipeline->getDescriptorHeap()->GetCPUHandleAt(0), cmdList);
	velocityBuffer.passUAVDataToGPU(*context, computePipeline->getDescriptorHeap()->GetCPUHandleAt(1), cmdList);

	// Create Vertex & Index Buffer
	auto circleData = generateCircle(0.05f, 32);
	indexCount = circleData.second.size();

	vertexBuffer = VertexBuffer(circleData.first, circleData.first.size() * sizeof(XMFLOAT3), sizeof(XMFLOAT3));
	vbv = vertexBuffer.passVertexDataToGPU(*context, cmdList);

	indexBuffer = IndexBuffer(circleData.second, circleData.second.size() * sizeof(unsigned int));
	ibv = indexBuffer.passIndexDataToGPU(*context, cmdList);

	//Transition both buffers to their usable states
	D3D12_RESOURCE_BARRIER barriers[2] = {};

	// Vertex buffer barrier
	barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[0].Transition.pResource = vertexBuffer.getVertexBuffer().Get();
	barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
	barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	// Index buffer barrier
	barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	barriers[1].Transition.pResource = indexBuffer.getIndexBuffer().Get();
	barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
	barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

	cmdList->ResourceBarrier(2, barriers);

	pipeline->createPSOD();
	pipeline->createPipelineState(context->getDevice());

	context->executeCommandList();

	context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
}

void PBMPMScene::compute() {

	context->initCommandList();

	D3D12_RESOURCE_BARRIER UAVbarrier = {};
	UAVbarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	UAVbarrier.Transition.pResource = positionBuffer.getBuffer();  // The resource used as UAV and SRV
	UAVbarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	UAVbarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	UAVbarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	cmdList->ResourceBarrier(1, &UAVbarrier);

	// Run command list
	context->executeCommandList();
	context->signalAndWaitForFence(fence, fenceValue);

	// --- Begin Compute Pass ---
	cmdList = context->initCommandList();
	cmdList->SetPipelineState(computePipeline->getPSO());
	cmdList->SetComputeRootSignature(computePipeline->getRootSignature());

	//// Set descriptor heap
	ID3D12DescriptorHeap* computeDescriptorHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

	//// Set compute root constants
	PBMPMConstants constants = { 9.81f, 1.0f, 0.0f, 0.0005f };

	cmdList->SetComputeRoot32BitConstants(0, 4, &constants, 0);

	//// Set compute root descriptor table
	cmdList->SetComputeRootDescriptorTable(1, computePipeline->getDescriptorHeap()->GetGPUHandleAt(0));

	//// Dispatch
	cmdList->Dispatch(instanceCount, 1, 1);

	//// Execute command list
	context->executeCommandList();
	context->signalAndWaitForFence(fence, fenceValue);

	// Reinitialize command list
	cmdList = context->initCommandList();

	// Transition position buffer to SRV
	D3D12_RESOURCE_BARRIER SRVbarrier = {};
	SRVbarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	SRVbarrier.Transition.pResource = positionBuffer.getBuffer();  // The resource used as UAV and SRV
	SRVbarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	SRVbarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	SRVbarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	cmdList->ResourceBarrier(1, &SRVbarrier);

	// --- End Compute Pass ---
	context->executeCommandList();
	context->signalAndWaitForFence(fence, fenceValue);
}

void PBMPMScene::draw(Camera* cam) {;
	// IA
	cmdList->IASetVertexBuffers(0, 1, &vbv);
	cmdList->IASetIndexBuffer(&ibv);
	cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// PSO
	cmdList->SetPipelineState(pipeline->getPSO());
	cmdList->SetGraphicsRootSignature(pipeline->getRootSignature());

	ID3D12DescriptorHeap* descriptorHeaps[] = { pipeline->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	auto viewMat = cam->getViewMat();
	auto projMat = cam->getProjMat();
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &modelMat, 32);
	cmdList->SetGraphicsRootShaderResourceView(1, positionBuffer.getBuffer()->GetGPUVirtualAddress()); // Descriptor table slot 1 for position SRV

	// Draw
	cmdList->DrawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);
}

void PBMPMScene::releaseResources() {
	positionBuffer.releaseResources();
	velocityBuffer.releaseResources();
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();
	computePipeline->releaseResources();
	pipeline->releaseResources();
}