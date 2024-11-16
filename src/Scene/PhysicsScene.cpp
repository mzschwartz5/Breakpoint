#include "PhysicsScene.h"

PhysicsScene::PhysicsScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* compPipeline, unsigned int instances)
	: Drawable(context, pipeline), context(context), pipeline(pipeline), 
	computePipeline(compPipeline), instanceCount(instances),
	modelMat(XMMatrixIdentity())
{
	constructScene();
}

void PhysicsScene::constructScene() {
	// Create Model Matrix
	modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

	// Create position and velocity data
	for (int i = 0; i < instanceCount; ++i) {
		positions.push_back({ -0.72f + 0.15f * i, 0.f, 1.f });
	}

	for (int i = 0; i < instanceCount; ++i) {
		velocities.push_back({ 0.0f, 0.0f, 0.0f });
	}

	// Create Structured Buffers
	positionBuffer = StructuredBuffer(positions.data(), instanceCount, sizeof(XMFLOAT3), computePipeline->getDescriptorHeap());
	velocityBuffer = StructuredBuffer(velocities.data(), instanceCount, sizeof(XMFLOAT3), computePipeline->getDescriptorHeap());

	auto computeId = computePipeline->getCommandListID();

	// Pass Structured Buffers to Compute Pipeline
	positionBuffer.passUAVDataToGPU(*context, computePipeline->getCommandList(), computeId);
	velocityBuffer.passUAVDataToGPU(*context, computePipeline->getCommandList(), computeId);

	// Create Vertex & Index Buffer
	auto circleData = generateCircle(0.05f, 32);
	indexCount = circleData.second.size();

	vertexBuffer = VertexBuffer(circleData.first, circleData.first.size() * sizeof(XMFLOAT3), sizeof(XMFLOAT3));
	vbv = vertexBuffer.passVertexDataToGPU(*context, pipeline->getCommandList());

	indexBuffer = IndexBuffer(circleData.second, circleData.second.size() * sizeof(unsigned int));
	ibv = indexBuffer.passIndexDataToGPU(*context, pipeline->getCommandList());

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

	pipeline->getCommandList()->ResourceBarrier(2, barriers);

	pipeline->createPSOD();
	pipeline->createPipelineState(context->getDevice());

	// Execute and reset render pipeline command list
	context->executeCommandList(pipeline->getCommandListID());
	context->resetCommandList(pipeline->getCommandListID());

	context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
}

void PhysicsScene::compute() {

	// --- Begin Compute Pass ---

	auto cmdList = computePipeline->getCommandList();

	cmdList->SetPipelineState(computePipeline->getPSO());
	cmdList->SetComputeRootSignature(computePipeline->getRootSignature());

	// Set descriptor heap
	ID3D12DescriptorHeap* computeDescriptorHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

	// Set compute root constants
	Constants constants = { 9.81f, 1.0f, 0.0f, 0.0005f };

	cmdList->SetComputeRoot32BitConstants(0, 4, &constants, 0);

	// Set compute root descriptor table
	// Uses position's descriptor handle instead of velocity since it was allocated first
	// The descriptor table is looking for two buffers so it will give the consecutive one after position (velocity) 
	cmdList->SetComputeRootDescriptorTable(1, positionBuffer.getGPUDescriptorHandle());

	//// Dispatch
	cmdList->Dispatch(instanceCount, 1, 1);

	//// Execute command list
	context->executeCommandList(computePipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);

	// Reinitialize command list
	context->resetCommandList(computePipeline->getCommandListID());

	// Ensure that the writes to the UAV are completing before the SRV is used in the rendering pipeline
	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	uavBarrier.UAV.pResource = positionBuffer.getBuffer(); // The UAV resource written by the compute shader
	cmdList->ResourceBarrier(1, &uavBarrier);

	// Transition position buffer to SRV to the rendering pipeline
	D3D12_RESOURCE_BARRIER srvBarrier = {};
	srvBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	srvBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	srvBarrier.Transition.pResource = positionBuffer.getBuffer();  // The resource used as UAV and SRV
	srvBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	srvBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
	srvBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	pipeline->getCommandList()->ResourceBarrier(1, &srvBarrier);

	// --- End Compute Pass ---
	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());
}

void PhysicsScene::draw(Camera* cam) {

	auto cmdList = pipeline->getCommandList();

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
	cmdList->SetGraphicsRootShaderResourceView(1, positionBuffer.getGPUVirtualAddress()); // Descriptor table slot 1 for position SRV

	// Draw
	cmdList->DrawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);

	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	uavBarrier.Transition.pResource = positionBuffer.getBuffer();  // The resource used as UAV and SRV
	uavBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
	uavBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	uavBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	cmdList->ResourceBarrier(1, &uavBarrier);

	// Run command list, wait for fence, and reset
	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());

}

void PhysicsScene::releaseResources() {
	positionBuffer.releaseResources();
	velocityBuffer.releaseResources();
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();
	computePipeline->releaseResources();
	pipeline->releaseResources();
}