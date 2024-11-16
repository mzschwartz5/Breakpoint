#include "PBMPMScene.h"

PBMPMScene::PBMPMScene(DXContext* context, RenderPipeline* pipeline, unsigned int instances)
	: Scene(context, pipeline), context(context), pipeline(pipeline), instanceCount(instances),
	modelMat(XMMatrixIdentity()),
	g2p2gPipeline("g2p2gRootSignature.cso", "g2p2gComputeShader.cso", *context, CommandListID::PBMPM_G2P2G_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitCountPipeline("bukkitCountRootSignature.cso", "bukkitCountComputeShader.cso", *context, CommandListID::PBMPM_BUKKITCOUNT_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitAllocatePipeline("bukkitAllocateRootSignature.cso", "bukkitAllocateComputeShader.cso", *context, CommandListID::PBMPM_BUKKITALLOCATE_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bukkitInsertPipeline("bukkitInsertRootSignature.cso", "bukkitInsertComputeShader.cso", *context, CommandListID::PBMPM_BUKKITINSERT_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)
{
	constructScene();
}

void PBMPMScene::createBukkitSystem() {
	int bukkitCountX = std::ceil(constants.gridSize.x / BukkitSize);
	int bukkitCountY = std::ceil(constants.gridSize.y / BukkitSize);

	std::vector<int> count;
	count.resize(bukkitCountX * bukkitCountY);
	bukkitSystem.countBuffer = StructuredBuffer(count.data(), count.size(), sizeof(int), bukkitCountPipeline.getDescriptorHeap());

	std::vector<int> count2;
	count2.resize(bukkitCountX * bukkitCountY);
	bukkitSystem.countBuffer2 = StructuredBuffer(count2.data(), count2.size(), sizeof(int), bukkitInsertPipeline.getDescriptorHeap());

	std::vector<int> particleData;
	particleData.resize(bukkitCountX * bukkitCountY);
	bukkitSystem.countBuffer2 = StructuredBuffer(particleData.data(), particleData.size(), sizeof(int), bukkitInsertPipeline.getDescriptorHeap());

	std::vector<int> dispatch = { 0, 1, 1, 0 };
	bukkitSystem.dispatch = StructuredBuffer(dispatch.data(), dispatch.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	std::vector<int> blankDispatch = { 0, 1, 1, 0 };
	bukkitSystem.blankDispatch = StructuredBuffer(blankDispatch.data(), blankDispatch.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	std::vector<int> allocator = { 0, 0, 0, 0 };
	bukkitSystem.particleAllocator = StructuredBuffer(allocator.data(), allocator.size(), sizeof(int), bukkitAllocatePipeline.getDescriptorHeap());

	std::vector<int> indexStart;
	indexStart.resize(bukkitCountX * bukkitCountY);
	bukkitSystem.indexStart = StructuredBuffer(indexStart.data(), indexStart.size(), sizeof(int), bukkitAllocatePipeline.getDescriptorHeap());

	std::vector<BukkitThreadData> threadData;
	threadData.resize(10 * bukkitCountX * bukkitCountY);
	bukkitSystem.indexStart = StructuredBuffer(threadData.data(), threadData.size(), sizeof(BukkitThreadData), bukkitAllocatePipeline.getDescriptorHeap());

	bukkitSystem.countX = bukkitCountX;
	bukkitSystem.countY = bukkitCountY;
	bukkitSystem.count = bukkitCountX * bukkitCountY;
	bukkitSystem.countBuffer.passUAVDataToGPU(*context, bukkitCountPipeline.getCommandList(), bukkitCountPipeline.getCommandListID());
	bukkitSystem.countBuffer2.passUAVDataToGPU(*context, bukkitInsertPipeline.getCommandList(), bukkitInsertPipeline.getCommandListID());
	bukkitSystem.particleData.passUAVDataToGPU(*context, bukkitInsertPipeline.getCommandList(), bukkitInsertPipeline.getCommandListID());
	bukkitSystem.dispatch.passCBVDataToGPU(*context);
	bukkitSystem.blankDispatch.passCBVDataToGPU(*context);
	bukkitSystem.particleAllocator.passCBVDataToGPU(*context);
	bukkitSystem.indexStart.passUAVDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());
	bukkitSystem.threadData.passUAVDataToGPU(*context, bukkitAllocatePipeline.getCommandList(), bukkitAllocatePipeline.getCommandListID());
}

void PBMPMScene::bukkitizeParticles() {
	//clear buffers

	//copy blank dispatch to dispatch (reset dispatch)

	bukkitCountPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 18, &constants, 0);
	bukkitCountPipeline.getCommandList()->SetComputeRootConstantBufferView(1, particleCount.getGPUVirtualAddress());
	bukkitCountPipeline.getCommandList()->SetComputeRootDescriptorTable(0, particleBuffer.getGPUDescriptorHandle());
	bukkitCountPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.countBuffer.getGPUDescriptorHandle());

	//execute indirectly <3

	auto bukkitDispatchSizeX = std::floor((bukkitSystem.countX + GridDispatchSize - 1) / GridDispatchSize);
	auto bukkitDispatchSizeY = std::floor((bukkitSystem.countY + GridDispatchSize - 1) / GridDispatchSize);

	bukkitAllocatePipeline.getCommandList()->SetComputeRoot32BitConstants(0, 18, &constants, 0);
	bukkitAllocatePipeline.getCommandList()->SetComputeRootUnorderedAccessView(0, bukkitSystem.countBuffer.getGPUVirtualAddress());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootUnorderedAccessView(1, bukkitSystem.dispatch.getGPUVirtualAddress());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(0, bukkitSystem.threadData.getGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.particleAllocator.getGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(2, bukkitSystem.indexStart.getGPUDescriptorHandle());

	//execute

	bukkitInsertPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 18, &constants, 0);
	bukkitInsertPipeline.getCommandList()->SetComputeRootConstantBufferView(1, particleCount.getGPUVirtualAddress());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(0, bukkitSystem.countBuffer2.getGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(1, particleBuffer.getGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(2, bukkitSystem.particleData.getGPUDescriptorHandle());
	bukkitAllocatePipeline.getCommandList()->SetComputeRootDescriptorTable(3, bukkitSystem.indexStart.getGPUDescriptorHandle());

	//execute
}

void PBMPMScene::constructScene() {
	auto computeId = g2p2gPipeline.getCommandListID();

	// Create Model Matrix
	modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

	float radius = 0.0005;
	float spacing = radius * 2.1;

	int particlesPerRow = (int)sqrt(instanceCount);
	int particlesPerCol = (instanceCount - 1) / particlesPerRow + 1;

	// Create position and velocity data
	for (int i = 0; i < instanceCount; ++i) {
		positions.push_back({ (i % particlesPerRow) * spacing - (particlesPerRow - 1) * spacing / 2.f,
							  (i / particlesPerRow) * spacing - (particlesPerCol - 1) * spacing / 2.f, 
							  0.f});
	}

	// Create Structured Buffers
	positionBuffer = StructuredBuffer(positions.data(), instanceCount, sizeof(XMFLOAT3), g2p2gPipeline.getDescriptorHeap());

	std::vector<PBMPMParticle> particles;
	particles.resize(maxParticles);
	particleBuffer = StructuredBuffer(particles.data(), particles.size(), sizeof(PBMPMParticle), g2p2gPipeline.getDescriptorHeap());
	
	std::vector<int> freeIndices;
	freeIndices.resize(1 + maxParticles); //maybe four maybe one idk
	particleFreeIndicesBuffer = StructuredBuffer(freeIndices.data(), freeIndices.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	std::vector<int> count = { 0, 0, 0, 0 };
	particleCount = StructuredBuffer(count.data(), count.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	//we prob dont need this
	std::vector<int> countStaging = { 0, 0, 0, 0 };
	particleCountStaging = StructuredBuffer(countStaging.data(), countStaging.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());
	
	std::vector<int> simDispatch = { 0, 1, 1, 0 };
	particleSimDispatch = StructuredBuffer(simDispatch.data(), simDispatch.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	//we prob dont need this
	std::vector<int> freeCountStaging = { 0, 0, 0, 0 };
	particleFreeCountStaging = StructuredBuffer(freeCountStaging.data(), freeCountStaging.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());

	// Pass Structured Buffers to Compute Pipeline
	positionBuffer.passUAVDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);

	particleBuffer.passUAVDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);
	particleFreeIndicesBuffer.passUAVDataToGPU(*context, g2p2gPipeline.getCommandList(), computeId);

	particleCount.passCBVDataToGPU(*context);
	particleSimDispatch.passCBVDataToGPU(*context);

	//buffers updated per frame
	bukkitSystem = BukkitSystem{};
	createBukkitSystem();

	std::vector<int> gridBufferData;
	gridBufferData.resize(constants.gridSize.x * constants.gridSize.y * 4);

	for (int i = 0; i < 3; i++) {
		gridBuffers[i] = StructuredBuffer(gridBufferData.data(), gridBufferData.size(), sizeof(int), g2p2gPipeline.getDescriptorHeap());
		gridBuffers[i].passUAVDataToGPU(*context, g2p2gPipeline.getCommandList(), g2p2gPipeline.getCommandListID());
	}

	// Create Vertex & Index Buffer
	auto circleData = generateCircle(radius, 32);
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

void PBMPMScene::compute() {

	int bufferIdx = 0;
	
	//update bukkit system

	//update grid buffers

	int substepCount = 20;
	for (int substepIdx = 0; substepIdx < substepCount; substepIdx++) {
		
		for (int iterationIdx = 0; iterationIdx < constants.iterationCount; iterationIdx++) {
			constants.iteration = iterationIdx;
			g2p2gPipeline.getCommandList()->SetComputeRoot32BitConstants(0, 18, &constants, 0);

			auto currentGrid = gridBuffers[bufferIdx];
			auto nextGrid = gridBuffers[(bufferIdx + 1) % 3];
			auto nextNextGrid = gridBuffers[(bufferIdx + 2) % 3];
			bufferIdx = (bufferIdx + 1) % 3;

			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(0, currentGrid.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(1, bukkitSystem.threadData.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(2, bukkitSystem.particleData.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(3, particleBuffer.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(4, nextGrid.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(5, nextNextGrid.getGPUDescriptorHandle());
			g2p2gPipeline.getCommandList()->SetComputeRootDescriptorTable(6, particleFreeIndicesBuffer.getGPUDescriptorHandle());
		}
	}

	// --- Begin Compute Pass ---

	auto cmdList = g2p2gPipeline.getCommandList();

	cmdList->SetPipelineState(g2p2gPipeline.getPSO());
	cmdList->SetComputeRootSignature(g2p2gPipeline.getRootSignature());

	//// Set descriptor heap
	ID3D12DescriptorHeap* computeDescriptorHeaps[] = { g2p2gPipeline.getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

	//// Set compute root constants
	constants = { {10, 10}, 0.0005, 9.81, 1.5, 0.05, 5, 1, 90, 0 };

	cmdList->SetComputeRoot32BitConstants(0, 9, &constants, 0);

	// Uses position's descriptor handle instead of velocity since it was allocated first
	// The descriptor table is looking for two buffers so it will give the consecutive one after position (velocity) 
	cmdList->SetComputeRootDescriptorTable(1, positionBuffer.getGPUDescriptorHandle());

	//// Dispatch
	cmdList->Dispatch(instanceCount, 1, 1);

	//// Execute command list
	context->executeCommandList(g2p2gPipeline.getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);

	// Reinitialize command list
	context->resetCommandList(g2p2gPipeline.getCommandListID());

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

void PBMPMScene::draw(Camera* cam) {;

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

void PBMPMScene::releaseResources() {
	positionBuffer.releaseResources();
	velocityBuffer.releaseResources();
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();
	g2p2gPipeline->releaseResources();
	pipeline->releaseResources();
}