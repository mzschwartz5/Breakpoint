#include "PBD.h"


PBDScene::PBDScene(DXContext* context, RenderPipeline* pipeline, 
	ComputePipeline* computePipeline, ComputePipeline* applyForcesPipeline, 
	ComputePipeline* velocityUpdatePipeline, ComputePipeline* FaceToFacePipeline,
	unsigned int instances)
	: Scene(context, pipeline), context(context), pipeline(pipeline),
	computePipeline(computePipeline), applyForcesPipeline(applyForcesPipeline), 
	velocityUpdatePipeline(velocityUpdatePipeline), 
	FaceToFacePipeline(FaceToFacePipeline),
	instanceCount(instances),
	modelMat(XMMatrixIdentity())
{
	constructScene();
}

//void PBDScene::testBreaking(std::vector<Particle> particles) {
//	// Apply force to second voxel to test breaking
//	for (int i = 8; i < 16; i++) {
//		particles[i].velocity = XMFLOAT3(3.0f, 0.0f, 0.0f); // Push second voxel right
//	}
//}
//
//void PBDScene::testTwisting(std::vector<Particle> particles) {
//	// Apply twist force to second voxel
//	for (int i = 8; i < 16; i++) {
//		particles[i].velocity = XMFLOAT3(0.0f, 1.0f * ((i % 2) * 2 - 1), 0.0f);
//	}
//}

void PBDScene::testVoxels(std::vector<Particle>* particles,
	std::vector<Voxel>* voxels) {

	std::vector<XMFLOAT3> positions = {
		// Voxel 0 (Base cube)
		{ -0.15f, -0.15f, -0.15f }, // 0
		{  0.15f, -0.15f, -0.15f }, // 1
		{  0.15f,  0.15f, -0.15f }, // 2
		{ -0.15f,  0.15f, -0.15f }, // 3
		{ -0.15f, -0.15f,  0.15f }, // 4
		{  0.15f, -0.15f,  0.15f }, // 5
		{  0.15f,  0.15f,  0.15f }, // 6
		{ -0.15f,  0.15f,  0.15f }, // 7

		// Voxel 1 (Right connected cube - gap of 0.06)
		{ 0.21f, -0.15f, -0.15f },  // 8
		{ 0.51f, -0.15f, -0.15f },  // 9
		{ 0.51f,  0.15f, -0.15f },  // 10
		{ 0.21f,  0.15f, -0.15f },  // 11
		{ 0.21f, -0.15f,  0.15f },  // 12
		{ 0.51f, -0.15f,  0.15f },  // 13
		{ 0.51f,  0.15f,  0.15f },  // 14
		{ 0.21f,  0.15f,  0.15f },  // 15

		// Voxel 2 (Stacked on top - gap of 0.06)
		{ -0.15f,  0.21f, -0.15f }, // 16
		{  0.15f,  0.21f, -0.15f }, // 17
		{  0.15f,  0.51f, -0.15f }, // 18
		{ -0.15f,  0.51f, -0.15f }, // 19
		{ -0.15f,  0.21f,  0.15f }, // 20
		{  0.15f,  0.21f,  0.15f }, // 21
		{  0.15f,  0.51f,  0.15f }, // 22
		{ -0.15f,  0.51f,  0.15f }, // 23


	};

	particles->resize(positions.size());
	for (size_t i = 0; i < positions.size(); ++i) {
		(*particles)[i].position = positions[i];
		(*particles)[i].prevPosition = positions[i];
		(*particles)[i].velocity = { 0.0f, 0.0f, 0.0f };
		(*particles)[i].invMass = 1.0f;
	}



	// Voxel 0 (Base)
	Voxel voxel0;
	for (int i = 0; i < 8; ++i) {
		voxel0.particleIndices[i] = i;
	}
	voxel0.u = { 1.0f, 0.0f, 0.0f };
	voxel0.v = { 0.0f, 1.0f, 0.0f };
	voxel0.w = { 0.0f, 0.0f, 1.0f };
	for (int i = 0; i < 6; i++) {
		voxel0.faceConnections[i] = true;
		voxel0.faceStrains[i] = 0.0f;
		voxel0.shapeLambda[i] = 0.0f;
	}

	// Voxel 1 (Right connected)
	Voxel voxel1;
	for (int i = 0; i < 8; ++i) {
		voxel1.particleIndices[i] = i + 8;
	}
	voxel1.u = { 1.0f, 0.0f, 0.0f };
	voxel1.v = { 0.0f, 1.0f, 0.0f };
	voxel1.w = { 0.0f, 0.0f, 1.0f };
	for (int i = 0; i < 6; i++) {
		voxel1.faceConnections[i] = true;
		voxel1.faceStrains[i] = 0.0f;
		voxel1.shapeLambda[i] = 0.0f;
	}

	// Voxel 2 (Stacked)
	Voxel voxel2;
	for (int i = 0; i < 8; ++i) {
		voxel2.particleIndices[i] = i + 16;
	}
	voxel2.u = { 1.0f, 0.0f, 0.0f };
	voxel2.v = { 0.0f, 1.0f, 0.0f };
	voxel2.w = { 0.0f, 0.0f, 1.0f };
	for (int i = 0; i < 6; i++) {
		voxel2.faceConnections[i] = true;
		voxel2.faceStrains[i] = 0.0f;
		voxel2.shapeLambda[i] = 0.0f;
	}



	// Setup voxel connections
	voxel0.faceConnections[0] = true;  // Connect to voxel1
	voxel1.faceConnections[1] = true;  // Connect to voxel0

	voxel0.faceConnections[2] = true;  // Connect to voxel2
	voxel2.faceConnections[3] = true;  // Connect to voxel0




	// Push voxels
	voxels->push_back(voxel0);
	voxels->push_back(voxel1);
	voxels->push_back(voxel2);


}

void PBDScene::createPartitions(
	std::vector<std::vector<uint32_t>>* partitionIndices,
	std::vector<Particle>* particles,
	std::vector<Voxel>* voxels, ComputePipeline* computePipeline) {


	partitionIndices->clear();
	partitionIndices->resize(4);

	for (uint32_t i = 0; i < voxels->size(); i++) {
		(*partitionIndices)[0].push_back(i);
	}


	// Partition 1: X-axis faces
	for (uint32_t i = 0; i < voxels->size(); i++) {
		if ((*voxels)[i].faceConnections[0] || (*voxels)[i].faceConnections[1]) {
			(*partitionIndices)[1].push_back(i);
		}
	}

	// Partition 2: Y-axis faces
	for (uint32_t i = 0; i < voxels->size(); i++) {
		if ((*voxels)[i].faceConnections[2] || (*voxels)[i].faceConnections[3]) {
			(*partitionIndices)[2].push_back(i);
		}
	}

	// Partition 3: Z-axis faces
	for (uint32_t i = 0; i < voxels->size(); i++) {
		if ((*voxels)[i].faceConnections[4] || (*voxels)[i].faceConnections[5]) {
			(*partitionIndices)[3].push_back(i);
		}
	}

	
	
}


void PBDScene::constructScene() {
	// Create Model Matrix
	modelMat *= XMMatrixTranslation(0.0f, 0.0f, 0.0f);

	

	testVoxels(&particles, &voxels);
	


	auto now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	float randomSeed = static_cast<float>(duration.count());


	simParams.deltaTime = 0.033f;
	simParams.count = instanceCount;
	simParams.gravity = { 0.0f, -9.81f, 0.0f };
	simParams.constraintCount = static_cast<unsigned int>(voxels.size());
	simParams.breakingThreshold = 0.4f;
	simParams.randomSeed = randomSeed;

	simParams.numSubsteps = 20.0f;  // As recommended in the paper
	simParams.compliance = 0.0001f;

	
	simParams.partitionSize = instanceCount/8.0f;
	
	createPartitions(&partitionIndices, &particles, &voxels, computePipeline);

	particleBuffer = StructuredBuffer(particles.data(), particles.size(),
		sizeof(Particle), computePipeline->getDescriptorHeap());
	voxelBuffer = StructuredBuffer(voxels.data(), voxels.size(),
		sizeof(Voxel), computePipeline->getDescriptorHeap());

	if (!partitionIndices[0].empty()) {
		shapePartitionBuffer = StructuredBuffer(
			partitionIndices[0].data(),
			partitionIndices[0].size(),
			sizeof(uint32_t),
			computePipeline->getDescriptorHeap()
		);
		
	}

	if (!partitionIndices[1].empty()) {
		xFacePartitionBuffer = StructuredBuffer(
			partitionIndices[1].data(),
			partitionIndices[1].size(),
			sizeof(uint32_t),
			computePipeline->getDescriptorHeap()
		);
		
	}
	if (!partitionIndices[2].empty()) {
		yFacePartitionBuffer = StructuredBuffer(
			partitionIndices[2].data(),
			partitionIndices[2].size(),
			sizeof(uint32_t),
			computePipeline->getDescriptorHeap()
		);
		
	}
	if (!partitionIndices[3].empty()) {
		zFacePartitionBuffer = StructuredBuffer(
			partitionIndices[3].data(),
			partitionIndices[3].size(),
			sizeof(uint32_t),
			computePipeline->getDescriptorHeap()
		);
		
	}
	

	auto computeList = computePipeline->getCommandList();
	auto forcesList = applyForcesPipeline->getCommandList();
	auto velocityList = velocityUpdatePipeline->getCommandList();

	

	particleBuffer.passUAVDataToGPU(*context, computeList, computePipeline->getCommandListID());
	voxelBuffer.passUAVDataToGPU(*context, computeList, computePipeline->getCommandListID());
	
	
	shapePartitionBuffer.passSRVDataToGPU(*context, computePipeline->getCommandList(), computePipeline->getCommandListID());
	xFacePartitionBuffer.passSRVDataToGPU(*context, computePipeline->getCommandList(), computePipeline->getCommandListID());
	yFacePartitionBuffer.passSRVDataToGPU(*context, computePipeline->getCommandList(), computePipeline->getCommandListID());
	zFacePartitionBuffer.passSRVDataToGPU(*context, computePipeline->getCommandList(), computePipeline->getCommandListID());
	

	
	

	// Create Vertex & Index Buffer
	auto  circleData = generateSphere(0.05f, 16, 16);
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
	float fullTimeStep = simParams.deltaTime;



	for (int substep = 0; substep < static_cast<int>(simParams.numSubsteps); substep++) {
		// Adjust deltaTime for substep
		simParams.deltaTime = fullTimeStep / simParams.numSubsteps;


		auto forcesList = applyForcesPipeline->getCommandList();
		//context->resetCommandList(CommandListID::apply_force_ID);


		forcesList->SetPipelineState(applyForcesPipeline->getPSO());
		forcesList->SetComputeRootSignature(applyForcesPipeline->getRootSignature());

		ID3D12DescriptorHeap* applyForcesHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
		forcesList->SetDescriptorHeaps(_countof(applyForcesHeaps), applyForcesHeaps);
		forcesList->SetComputeRootDescriptorTable(1, particleBuffer.getGPUDescriptorHandle());
		forcesList->SetComputeRoot32BitConstants(0, sizeof(SimulationParams) / 4, &simParams, 0);
		forcesList->Dispatch(instanceCount, 1, 1);

		// UAV barrier to ensure forces are applied before constraints
		D3D12_RESOURCE_BARRIER uavBarrier = {};
		uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		uavBarrier.UAV.pResource = particleBuffer.getBuffer();
		forcesList->ResourceBarrier(1, &uavBarrier);


		context->executeCommandList(applyForcesPipeline->getCommandListID());
		context->signalAndWaitForFence(fence, ++fenceValue);
		
		if (!partitionIndices[0].empty()) {

			simParams.partitionSize = static_cast<uint32_t>(partitionIndices[0].size());
			auto constraintList = computePipeline->getCommandList();

			constraintList->SetPipelineState(computePipeline->getPSO());
			constraintList->SetComputeRootSignature(computePipeline->getRootSignature());
			ID3D12DescriptorHeap* constraintHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
			constraintList->SetDescriptorHeaps(_countof(constraintHeaps), constraintHeaps);

			constraintList->SetComputeRootDescriptorTable(0, computePipeline->getDescriptorHeap()->GetGPUHandleAt(0)); // Points to first UAV
			constraintList->SetComputeRoot32BitConstants(1, sizeof(SimulationParams) / 4, &simParams, 0);
			constraintList->SetComputeRootDescriptorTable(2, computePipeline->getDescriptorHeap()->GetGPUHandleAt(2)); // Points to SRV


			UINT numThreadGroups = (simParams.partitionSize + 255) / 256;
			constraintList->Dispatch(numThreadGroups, 1, 1);

			context->executeCommandList(computePipeline->getCommandListID());
			context->signalAndWaitForFence(fence, ++fenceValue);
		}
	
		for (int i = 1; i < 4; i++) {
			if (!partitionIndices[i].empty()) {
				simParams.partitionSize = static_cast<uint32_t>(partitionIndices[i].size());
				auto Face_ConstraintList = FaceToFacePipeline->getCommandList();
				Face_ConstraintList->SetComputeRootSignature(FaceToFacePipeline->getRootSignature());
				ID3D12DescriptorHeap* FaceToFaceHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
				Face_ConstraintList->SetDescriptorHeaps(_countof(FaceToFaceHeaps), FaceToFaceHeaps);

				Face_ConstraintList->SetComputeRootDescriptorTable(0, computePipeline->getDescriptorHeap()->GetGPUHandleAt(0));
				Face_ConstraintList->SetComputeRoot32BitConstants(1, sizeof(SimulationParams) / 4, &simParams, 0);

				StructuredBuffer* currentBuffer = nullptr;
				
				switch (i) {
				case 1:
					Face_ConstraintList->SetComputeRootDescriptorTable(2, computePipeline->getDescriptorHeap()->GetGPUHandleAt(3));
					
					break;
				case 2:
					
					Face_ConstraintList->SetComputeRootDescriptorTable(2, computePipeline->getDescriptorHeap()->GetGPUHandleAt(4));
					break;
				case 3:
					Face_ConstraintList->SetComputeRootDescriptorTable(2, computePipeline->getDescriptorHeap()->GetGPUHandleAt(5));
					
					break;
				default:
					continue;
				}

				//Face_ConstraintList->SetComputeRootDescriptorTable(2, currentBuffer->getGPUDescriptorHandle());


				UINT numThreadGroups = (simParams.partitionSize + 255) / 256;
				Face_ConstraintList->Dispatch(numThreadGroups, 1, 1);

				if (i < 3) {
					D3D12_RESOURCE_BARRIER barriers[2] = {};

					// Barrier for particle buffer
					barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
					barriers[0].UAV.pResource = particleBuffer.getBuffer();

					// Barrier for voxel buffer
					barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
					barriers[1].UAV.pResource = voxelBuffer.getBuffer();

					Face_ConstraintList->ResourceBarrier(2, barriers);
				}


				context->executeCommandList(FaceToFacePipeline->getCommandListID());
				context->signalAndWaitForFence(fence, ++fenceValue);
			}
		
		}
	

		auto velocityList = velocityUpdatePipeline->getCommandList();

		velocityList->SetPipelineState(velocityUpdatePipeline->getPSO());
		velocityList->SetComputeRootSignature(velocityUpdatePipeline->getRootSignature());
		ID3D12DescriptorHeap* velocityHeaps[] = { computePipeline->getDescriptorHeap()->Get() };
		velocityList->SetDescriptorHeaps(_countof(velocityHeaps), velocityHeaps);
		velocityList->SetComputeRootDescriptorTable(1, particleBuffer.getGPUDescriptorHandle());
		velocityList->SetComputeRoot32BitConstants(0, sizeof(SimulationParams) / 4, &simParams, 0);
		velocityList->Dispatch(instanceCount, 1, 1);


		D3D12_RESOURCE_BARRIER velocityBarrier = {};
		velocityBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		velocityBarrier.UAV.pResource = particleBuffer.getBuffer();
		velocityList->ResourceBarrier(1, &velocityBarrier);

		////velocityList->Close();
		context->executeCommandList(velocityUpdatePipeline->getCommandListID());
		context->signalAndWaitForFence(fence, ++fenceValue);




		// --- End Compute Pass ---
		context->resetCommandList(applyForcesPipeline->getCommandListID());
		context->resetCommandList(velocityUpdatePipeline->getCommandListID());
		context->resetCommandList(computePipeline->getCommandListID());
		context->resetCommandList(FaceToFacePipeline->getCommandListID());
	
	}
	simParams.deltaTime = fullTimeStep;

	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());
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
	cmdList->SetGraphicsRootUnorderedAccessView(1, particleBuffer.getGPUVirtualAddress()); // Descriptor table slot 1 for position SRV

	// Draw
	cmdList->DrawIndexedInstanced(indexCount, instanceCount, 0, 0, 0);


	// Run command list, wait for fence, and reset
	context->executeCommandList(pipeline->getCommandListID());
	context->signalAndWaitForFence(fence, fenceValue);
	context->resetCommandList(pipeline->getCommandListID());

}

void PBDScene::releaseResources() {
	particleBuffer.releaseResources();
	//constraintBuffer.releaseResources();
	vertexBuffer.releaseResources();
	indexBuffer.releaseResources();

	computePipeline->releaseResources();
	pipeline->releaseResources();
	velocityUpdatePipeline->releaseResources();
	applyForcesPipeline->releaseResources();
	FaceToFacePipeline->releaseResources();

	shapePartitionBuffer.releaseResources();
	xFacePartitionBuffer.releaseResources();
	yFacePartitionBuffer.releaseResources();
	zFacePartitionBuffer.releaseResources();
}