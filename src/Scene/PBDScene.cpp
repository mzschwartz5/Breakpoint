#include "PBD.h"


PBDScene::PBDScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* computePipeline, ComputePipeline* applyForcesPipeline, ComputePipeline* velocityUpdatePipeline,  unsigned int instances)
	: Scene(context, pipeline), context(context), pipeline(pipeline),
	computePipeline(computePipeline), applyForcesPipeline(applyForcesPipeline), velocityUpdatePipeline(velocityUpdatePipeline), instanceCount(instances),
	modelMat(XMMatrixIdentity())
{
	constructScene();
}

void PBDScene::constructScene() {
	// Create Model Matrix
	modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

	

	std::vector<XMFLOAT3> positions = {
	{ -0.1f, 0.0f, 0.0f }, // Particle 0
	{  0.1f, 0.0f, 0.0f }  // Particle 1
	};

	particles = std::vector<Particle>(positions.size());
	for (int i = 0; i < positions.size(); ++i) {
		particles[i].position = positions[i];
		particles[i].prevPosition = particles[i].position;
		particles[i].velocity = { 0.0f, 0.0f, 0.0f };
		particles[i].invMass = 1.0f; // Uniform mass
	}

	constraints = std::vector<DistanceConstraint>(1);
	constraints[0].particleA = 0;
	constraints[0].particleB = 1;
	constraints[0].restLength = 1.5f;


	
	simParams.deltaTime = 0.033f;
	simParams.count = instanceCount;
	simParams.gravity = { 0.0f, -9.81f, 0.0f };
	constraintCount = constraints.size();
	simParams.constraintCount = constraintCount;


	

	


	particleBuffer = StructuredBuffer(particles.data(), particles.size(),
		sizeof(Particle), computePipeline->getDescriptorHeap());
	constraintBuffer = StructuredBuffer(constraints.data(), constraints.size(),
		sizeof(DistanceConstraint), computePipeline->getDescriptorHeap());

	auto computeList = computePipeline->getCommandList();
	auto forcesList = applyForcesPipeline->getCommandList();
	auto velocityList = velocityUpdatePipeline->getCommandList();

	

	particleBuffer.passUAVDataToGPU(*context, computeList, computePipeline->getCommandListID());
	constraintBuffer.passSRVDataToGPU(*context, computeList, computePipeline->getCommandListID());
	
	/*particleBuffer.passUAVDataToGPU(*context, forcesList, applyForcesPipeline->getCommandListID());
	particleBuffer.passUAVDataToGPU(*context, velocityList, velocityUpdatePipeline->getCommandListID());*/

	//particleBuffer.passSRVDataToGPU(*context, computeList, computePipeline->getCommandListID());
	

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

void PBDScene::compute() {

	// --- Begin Compute Pass ---

	auto forcesList = applyForcesPipeline->getCommandList();
	//context->resetCommandList(CommandListID::apply_force_ID);
	

	forcesList->SetPipelineState(applyForcesPipeline->getPSO());
	forcesList->SetComputeRootSignature(applyForcesPipeline->getRootSignature());

	ID3D12DescriptorHeap* applyForcesHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
	forcesList->SetDescriptorHeaps(_countof(applyForcesHeaps), applyForcesHeaps);
	forcesList->SetComputeRootDescriptorTable(1, particleBuffer.getGPUDescriptorHandle());
	forcesList->SetComputeRoot32BitConstants(0, 5, &simParams, 0);
	forcesList->Dispatch(instanceCount, 1, 1);

	// UAV barrier to ensure forces are applied before constraints
	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	uavBarrier.UAV.pResource = particleBuffer.getBuffer();
	forcesList->ResourceBarrier(1, &uavBarrier);

	//forcesList->Close();
	context->executeCommandList(applyForcesPipeline->getCommandListID());
	context->signalAndWaitForFence(fence, ++fenceValue);

	auto constraintList = computePipeline->getCommandList();
	//context->resetCommandList(CommandListID::PBD_ID);

	constraintList->SetPipelineState(computePipeline->getPSO());
	constraintList->SetComputeRootSignature(computePipeline->getRootSignature());
	ID3D12DescriptorHeap* constraintHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
	constraintList->SetDescriptorHeaps(_countof(constraintHeaps), constraintHeaps);

	constraintList->SetComputeRootDescriptorTable(0, computePipeline->getDescriptorHeap()->GetGPUHandleAt(0));
	constraintList->SetComputeRootDescriptorTable(1, computePipeline->getDescriptorHeap()->GetGPUHandleAt(1));
	constraintList->SetComputeRoot32BitConstants(2, 1, &constraintCount, 0);

	for (int i = 0; i < 4; ++i) { 
		constraintList->Dispatch(constraintCount, 1, 1);

		// Add UAV barrier between iterations
		D3D12_RESOURCE_BARRIER iterationBarrier = {};
		iterationBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		iterationBarrier.UAV.pResource = particleBuffer.getBuffer();
		constraintList->ResourceBarrier(1, &iterationBarrier);
	}
	//constraintList->Close();
	context->executeCommandList(computePipeline->getCommandListID());
	context->signalAndWaitForFence(fence, ++fenceValue);

	auto velocityList = velocityUpdatePipeline->getCommandList();
	//context->resetCommandList(CommandListID::velocity_update_ID);

	velocityList->SetPipelineState(velocityUpdatePipeline->getPSO());
	velocityList->SetComputeRootSignature(velocityUpdatePipeline->getRootSignature());
	ID3D12DescriptorHeap* velocityHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
	velocityList->SetDescriptorHeaps(_countof(velocityHeaps), velocityHeaps);
	velocityList->SetComputeRootDescriptorTable(1, particleBuffer.getGPUDescriptorHandle());
	velocityList->SetComputeRoot32BitConstants(0, 5, &simParams, 0);
	velocityList->Dispatch(instanceCount, 1, 1);


	D3D12_RESOURCE_BARRIER velocityBarrier = {};
	velocityBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	velocityBarrier.UAV.pResource = particleBuffer.getBuffer();
	velocityList->ResourceBarrier(1, &velocityBarrier);

	//velocityList->Close();
	context->executeCommandList(velocityUpdatePipeline->getCommandListID());
	context->signalAndWaitForFence(fence, ++fenceValue);

	// Transition position buffer to SRV to the rendering pipeline
	//D3D12_RESOURCE_BARRIER srvBarrier = {};
	//srvBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	//srvBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
	//srvBarrier.Transition.pResource = particleBuffer.getBuffer();  // The resource used as UAV and SRV
	//srvBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	//srvBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
	//srvBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	//pipeline->getCommandList()->ResourceBarrier(1, &srvBarrier);



	// --- End Compute Pass ---
	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());
	context->resetCommandList(applyForcesPipeline->getCommandListID());
	context->resetCommandList(velocityUpdatePipeline->getCommandListID());
	context->resetCommandList(computePipeline->getCommandListID());

}

void PBDScene::draw(Camera* cam) {

	auto cmdList = pipeline->getCommandList();

	// IA
	cmdList->IASetVertexBuffers(0, 1, &vbv);
	cmdList->IASetIndexBuffer(&ibv);
	cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	// PSO

	D3D12_RESOURCE_BARRIER uavBarrier = {};
	uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	uavBarrier.Transition.pResource = particleBuffer.getBuffer();  // The resource used as UAV and SRV
	uavBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
	uavBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	uavBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	cmdList->ResourceBarrier(1, &uavBarrier);



	cmdList->SetPipelineState(pipeline->getPSO());
	cmdList->SetGraphicsRootSignature(pipeline->getRootSignature());

	ID3D12DescriptorHeap* descriptorHeaps[] = { pipeline->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
	auto viewMat = cam->getViewMat();
	auto projMat = cam->getProjMat();
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);
	cmdList->SetGraphicsRoot32BitConstants(0, 16, &modelMat, 32);
	cmdList->SetGraphicsRootShaderResourceView(1, particleBuffer.getGPUVirtualAddress()); // Descriptor table slot 1 for position SRV

	// Draw
	cmdList->DrawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);


	// Run command list, wait for fence, and reset
	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());

}

void PBDScene::releaseResources() {
	particleBuffer.releaseResources();
	constraintBuffer.releaseResources();
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();

	computePipeline->releaseResources();
	pipeline->releaseResources();
	velocityUpdatePipeline->releaseResources();
	applyForcesPipeline->releaseResources();
}