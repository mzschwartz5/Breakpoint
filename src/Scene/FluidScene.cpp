#include "FluidScene.h"

FluidScene::FluidScene(DXContext* context, 
                       RenderPipeline* pipeline, 
                       ComputePipeline* bilevelUniformGridCP, 
                       ComputePipeline* surfaceBlockDetectionCP,
                       ComputePipeline* surfaceCellDetectionCP,
                       ComputePipeline* surfaceVertexCompactionCP,
                       ComputePipeline* surfaceVertexDensityCP,
                       ComputePipeline* surfaceVertexNormalCP,
                       ComputePipeline* bufferClearCP,
                       ComputePipeline* dispatchArgDivideCP,
                       MeshPipeline* fluidMeshPipeline)
    : Drawable(context, pipeline), 
      bilevelUniformGridCP(bilevelUniformGridCP), 
      surfaceBlockDetectionCP(surfaceBlockDetectionCP),
      surfaceCellDetectionCP(surfaceCellDetectionCP),
      surfaceVertexCompactionCP(surfaceVertexCompactionCP),
      surfaceVertexDensityCP(surfaceVertexDensityCP),
      surfaceVertexNormalCP(surfaceVertexNormalCP),
      bufferClearCP(bufferClearCP),
      dispatchArgDivideCP(dispatchArgDivideCP),
      fluidMeshPipeline(fluidMeshPipeline)
{
    constructScene();
}

// In this pipeline, drawing is done via a mesh shader
void FluidScene::draw(
    Camera* camera,
    D3D12_GPU_DESCRIPTOR_HANDLE objectColorTextureHandle,
    D3D12_GPU_DESCRIPTOR_HANDLE objectPositionTextureHandle,
    int screenWidth,
    int screenHeight
) {
    timingFrame++;
    auto cmdList = fluidMeshPipeline->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    MeshShadingConstants meshShadingConstants = { 
        camera->getViewProjMat(), 
        gridConstants.gridDim, 
        gridConstants.resolution, 
        gridConstants.minBounds, 
        screenWidth, 
        camera->getPosition(), 
        screenHeight 
    };
    cmdList->SetPipelineState(fluidMeshPipeline->getPSO());
    cmdList->SetGraphicsRootSignature(fluidMeshPipeline->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* descriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get(), fluidMeshPipeline->getSamplerHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    // Transition surfaceHalfBlockDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertexNormalBuffer to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertexNormalBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexNormalBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to a UAV (this is purely for resetting the buffer)
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    
    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceVertDensityDispatchBarrier, surfaceHalfBlockDispatchBarrier, surfaceVertexNormalBufferBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set graphics root descriptor table
    cmdList->SetGraphicsRootDescriptorTable(0, surfaceBlockIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootDescriptorTable(1, surfaceVertDensityBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootDescriptorTable(2, surfaceVertexNormalBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetGraphicsRootDescriptorTable(3, objectColorTextureHandle);
    cmdList->SetGraphicsRootDescriptorTable(4, objectPositionTextureHandle);
    cmdList->SetGraphicsRootShaderResourceView(5, surfaceHalfBlockDispatch.getGPUVirtualAddress());
    cmdList->SetGraphicsRootUnorderedAccessView(6, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetGraphicsRootDescriptorTable(7, fluidMeshPipeline->getSamplerHandle());
    cmdList->SetGraphicsRoot32BitConstants(8, 28, &meshShadingConstants, 0);

    // Transition surfaceHalfBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceHalfBlockDispatchBarrier2);

    // Draws
    cmdList->ExecuteIndirect(meshCommandSignature, 1, surfaceHalfBlockDispatch.getBuffer(), 0, nullptr, 0);
    context->endTimingQuery(cmdList); // stop timer

    // TODO Temporary: just so these two buffers can be transitioned along with everything else
    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier3 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers2[2] = { surfaceHalfBlockDispatchBarrier3, surfaceVertDensityDispatchBarrier2 };
    cmdList->ResourceBarrier(2, barriers2);
    // End temporary

    transitionBuffers(cmdList, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    context->executeCommandList(fluidMeshPipeline->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(fluidMeshPipeline->getCommandListID());
    
    resetBuffers();
    
    meshShadingCumulativeTime += context->readTimingQueryData();
}

void FluidScene::constructScene() {
    gridConstants = {   
        static_cast<int>(AlembicLoader::getInstance().getMaxParticleCount()), 
        {BLOCKS_PER_EDGE * CELLS_PER_BLOCK_EDGE, BLOCKS_PER_EDGE * CELLS_PER_BLOCK_EDGE, BLOCKS_PER_EDGE * CELLS_PER_BLOCK_EDGE}, 
        {-8.0f, -8.0f, -8.0f}, 
        CELL_RESOLUTION
    };

    // Create cells and blocks buffers
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    
    std::vector<int> cellParticleCounts(numCells, 0);
    std::vector<int> cellParticleIndices(numCells * MAX_PARTICLES_PER_CELL, -1);
    std::vector<Block> blocks(numBlocks);
    std::vector<int> surfaceBlockIndices(numBlocks, 0);
    XMUINT3 dipatchCPU = { 0, 1, 1 };
    std::vector<int> surfaceVertices(numVerts, 0);
    std::vector<int> surfaceVertexIndices(numVerts, 0);
    std::vector<float> surfaceVertexDensities(numVerts, 0.f);
    std::vector<XMFLOAT3> surfaceVertexNormals(numVerts, { 0.f, 0.f, 0.f });

    positionBuffer = StructuredBuffer(AlembicLoader::getInstance().getParticlesForNextFrame(), AlembicLoader::getInstance().getMaxParticleCount(), sizeof(XMFLOAT3));
    positionBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    positionBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    // Use the descriptor heap for the bilevelUniformGridCP for pretty much everything. Simplifies sharing resources
    blocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block));
    blocksBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    blocksBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    blocksBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    cellParticleCountBuffer = StructuredBuffer(cellParticleCounts.data(), numCells, sizeof(int));
    cellParticleCountBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    cellParticleCountBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    cellParticleCountBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    cellParticleIndicesBuffer = StructuredBuffer(cellParticleIndices.data(), MAX_PARTICLES_PER_CELL * numCells, sizeof(int));
    cellParticleIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    cellParticleIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    cellParticleIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceBlockIndicesBuffer = StructuredBuffer(surfaceBlockIndices.data(), numBlocks, sizeof(unsigned int));
    surfaceBlockIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceBlockIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceBlockIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceBlockDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceBlockDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceBlockDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceBlockDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceHalfBlockDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceHalfBlockDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceHalfBlockDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceHalfBlockDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVerticesBuffer = StructuredBuffer(surfaceVertices.data(), numVerts, sizeof(unsigned int));
    surfaceVerticesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVerticesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVerticesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertexIndicesBuffer = StructuredBuffer(surfaceVertexIndices.data(), numVerts, sizeof(unsigned int));
    surfaceVertexIndicesBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertexIndicesBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertexIndicesBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertDensityDispatch = StructuredBuffer(&dipatchCPU, 1, sizeof(XMUINT3));
    surfaceVertDensityDispatch.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertDensityDispatch.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertDensityDispatch.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertDensityBuffer = StructuredBuffer(surfaceVertexDensities.data(), numVerts, sizeof(float));
    surfaceVertDensityBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertDensityBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap());
    surfaceVertDensityBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

    surfaceVertexNormalBuffer = StructuredBuffer(surfaceVertexNormals.data(), numVerts, sizeof(XMFLOAT3));
    surfaceVertexNormalBuffer.passDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());
    surfaceVertexNormalBuffer.createUAV(*context, bilevelUniformGridCP->getDescriptorHeap()); 
    surfaceVertexNormalBuffer.createSRV(*context, bilevelUniformGridCP->getDescriptorHeap());

	// Create Command Signature
	// Describe the arguments for an indirect dispatch
	D3D12_INDIRECT_ARGUMENT_DESC argumentDesc = {};
	argumentDesc.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

	// Command signature description
	D3D12_COMMAND_SIGNATURE_DESC commandSignatureDesc = {};
	commandSignatureDesc.ByteStride = sizeof(XMUINT3); // Argument buffer stride
	commandSignatureDesc.NumArgumentDescs = 1; // One argument descriptor
	commandSignatureDesc.pArgumentDescs = &argumentDesc;

	// Create the command signature
	context->getDevice()->CreateCommandSignature(&commandSignatureDesc, nullptr, IID_PPV_ARGS(&commandSignature));

    // Command signature for mesh shader
    D3D12_INDIRECT_ARGUMENT_DESC argumentDescMesh = {};
    argumentDescMesh.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;

    D3D12_COMMAND_SIGNATURE_DESC commandSignatureDescMesh = {};
    commandSignatureDescMesh.ByteStride = sizeof(XMUINT3); // Argument buffer stride
    commandSignatureDescMesh.NumArgumentDescs = 1; // One argument descriptor
    commandSignatureDescMesh.pArgumentDescs = &argumentDescMesh;

    context->getDevice()->CreateCommandSignature(&commandSignatureDescMesh, nullptr, IID_PPV_ARGS(&meshCommandSignature));

    // Create fence
    context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Transition all resources to UAVs to start
    transitionBuffers(bilevelUniformGridCP->getCommandList(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    context->executeCommandList(bilevelUniformGridCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);
    context->resetCommandList(bilevelUniformGridCP->getCommandListID());
}

void FluidScene::compute() {
    positionBuffer.updateDataOnGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID(), AlembicLoader::getInstance().getParticlesForNextFrame());

    gridConstants.numParticles = AlembicLoader::getInstance().getParticleCountForCurrentFrame();

    computeBilevelUniformGrid();
    computeSurfaceBlockDetection();
    computeSurfaceCellDetection();
    compactSurfaceVertices();
    computeSurfaceVertexDensity();
    computeSurfaceVertexNormal();

    printTimings();
}

void FluidScene::printTimings() {
    if (timingFrame % 1000 != 0) return;

    double computeTimeAvg;

    computeTimeAvg = bilevelGridCumulativeTime / 1000.0;
    bilevelGridCumulativeTime = 0.0;
    printf("BilevelUniformGrid Time: %.3f ms\n", computeTimeAvg);

    computeTimeAvg = blockDetectionCumulativeTime / 1000.0;
    blockDetectionCumulativeTime = 0.0;
    printf("SurfaceBlockDetection Time: %.3f ms\n", computeTimeAvg);

    computeTimeAvg = cellDetectionCumulativeTime / 1000.0;
    cellDetectionCumulativeTime = 0.0;
    printf("SurfaceCellDetection Time: %.3f ms\n", computeTimeAvg);
    
    computeTimeAvg = compactSurfVertsCumulativeTime / 1000.0;
    compactSurfVertsCumulativeTime = 0.0;
    printf("CompactSurfaceVerts Time: %.3f ms\n", computeTimeAvg);

    computeTimeAvg = surfVertDensityCumulativeTime / 1000.0;
    surfVertDensityCumulativeTime = 0.0;
    printf("SurfaceVertexDensity Time: %.3f ms\n", computeTimeAvg);

    computeTimeAvg = surfVertNormalCumulativeTime / 1000.0;
    surfVertNormalCumulativeTime = 0.0;
    printf("SurfaceVertexNormal Time: %.3f ms\n", computeTimeAvg);

    computeTimeAvg = meshShadingCumulativeTime / 1000.0;
    meshShadingCumulativeTime = 0.0;
    printf("MeshShading Time: %.3f ms\n", computeTimeAvg);
}

void FluidScene::computeBilevelUniformGrid() {
    auto cmdList = bilevelUniformGridCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(bilevelUniformGridCP->getPSO());
    cmdList->SetComputeRootSignature(bilevelUniformGridCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, cellParticleIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(3, blocksBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(4, 8, &gridConstants, 0);

    // Dispatch
    int numWorkGroups = (gridConstants.numParticles + BILEVEL_UNIFORM_GRID_THREADS_X - 1) / BILEVEL_UNIFORM_GRID_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);
    context->endTimingQuery(cmdList); // stop timer

    // Transition blocksBuffer from UAV to SRV for the next pass
    D3D12_RESOURCE_BARRIER blocksBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        blocksBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    cmdList->ResourceBarrier(1, &blocksBufferBarrier);

    // Execute command list
    context->executeCommandList(bilevelUniformGridCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    // Reinitialize command list
    context->resetCommandList(bilevelUniformGridCP->getCommandListID());
    bilevelGridCumulativeTime += context->readTimingQueryData();
}

void FluidScene::computeSurfaceBlockDetection() {
    auto cmdList = surfaceBlockDetectionCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(surfaceBlockDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceBlockDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, blocksBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceBlockIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 1, &numBlocks, 0);

    // Dispatch
    int numWorkGroups = (numBlocks + SURFACE_BLOCK_DETECTION_THREADS_X - 1) / SURFACE_BLOCK_DETECTION_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);
    context->endTimingQuery(cmdList); // stop timer

    context->executeCommandList(surfaceBlockDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceBlockDetectionCP->getCommandListID());
    blockDetectionCumulativeTime += context->readTimingQueryData();
}

void FluidScene::computeSurfaceCellDetection() {
    auto cmdList = surfaceCellDetectionCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(surfaceCellDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceCellDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceBlockIndicesBuffer to SRV 
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceBlockDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition cellsBuffer to SRV 
    D3D12_RESOURCE_BARRIER cellParticleCountBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleCountBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceBlockIndicesBufferBarrier, surfaceBlockDispatchBarrier, cellParticleCountBufferBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceBlockIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVerticesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(3, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(4, surfaceHalfBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(5, 3, &gridConstants.gridDim, 0);

    // Transition surfaceBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceBlockDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceBlockDispatch.getBuffer(), 0, nullptr, 0);
    context->endTimingQuery(cmdList); // stop timer

    context->executeCommandList(surfaceCellDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceCellDetectionCP->getCommandListID());

    cellDetectionCumulativeTime += context->readTimingQueryData();
}

void FluidScene::compactSurfaceVertices() {
    auto cmdList = surfaceVertexCompactionCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(surfaceVertexCompactionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexCompactionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVerticesBuffer back to SRV 
    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE 
    );

    cmdList->ResourceBarrier(1, &surfaceVerticesBufferBarrier);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceVerticesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 3, &gridConstants.gridDim, 0);

    // Dispatch
    int numVertices = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    int numWorkGroups = (numVertices + SURFACE_VERTEX_COMPACTION_THREADS_X - 1) / SURFACE_VERTEX_COMPACTION_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);
    context->endTimingQuery(cmdList); // stop timer

    context->executeCommandList(surfaceVertexCompactionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexCompactionCP->getCommandListID());

    divNumThreadsByGroupSize(&surfaceVertDensityDispatch, SURFACE_VERTEX_DENSITY_THREADS_X);

    compactSurfVertsCumulativeTime += context->readTimingQueryData();
}

void FluidScene::computeSurfaceVertexDensity() {
    auto cmdList = surfaceVertexDensityCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(surfaceVertexDensityCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexDensityCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVertexIndicesBuffer to SRV
    D3D12_RESOURCE_BARRIER surfaceVertexIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceBlockDispatch to a UAV
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    D3D12_RESOURCE_BARRIER barriers[3] = { surfaceVertexIndicesBufferBarrier, surfaceVertDensityDispatchBarrier, surfaceBlockDispatchBarrier };
    cmdList->ResourceBarrier(3, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, cellParticleIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(3, surfaceVertexIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(4, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootUnorderedAccessView(5, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(6, surfaceVertDensityBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(7, 8, &gridConstants, 0);

    // Transition surfaceVertDensityDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceVertDensityDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceVertDensityDispatch.getBuffer(), 0, nullptr, 0);
    context->endTimingQuery(cmdList); // stop timer

    context->executeCommandList(surfaceVertexDensityCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexDensityCP->getCommandListID());
    surfVertDensityCumulativeTime += context->readTimingQueryData();
}

void FluidScene::computeSurfaceVertexNormal() {
    auto cmdList = surfaceVertexNormalCP->getCommandList();
    context->startTimingQuery(cmdList); // start timer

    cmdList->SetPipelineState(surfaceVertexNormalCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceVertexNormalCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Transition surfaceVertDensityBuffer to SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceVertDensityDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[2] = { surfaceVertDensityBufferBarrier, surfaceVertDensityDispatchBarrier };
    cmdList->ResourceBarrier(2, barriers);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceVertDensityBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getSRVGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(2, surfaceVertDensityDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRootDescriptorTable(3, surfaceVertexNormalBuffer.getUAVGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(4, 8, &gridConstants, 0);

    // Transition surfaceVertDensityDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceVertDensityDispatchBarrier2);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceVertDensityDispatch.getBuffer(), 0, nullptr, 0);
    context->endTimingQuery(cmdList); // stop timer

    context->executeCommandList(surfaceVertexNormalCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceVertexNormalCP->getCommandListID());
    surfVertNormalCumulativeTime += context->readTimingQueryData();
}

void FluidScene::releaseResources() {
    renderPipeline->releaseResources();
    bilevelUniformGridCP->releaseResources();
    surfaceBlockDetectionCP->releaseResources();
    surfaceCellDetectionCP->releaseResources();
    cellParticleCountBuffer.releaseResources();
    cellParticleIndicesBuffer.releaseResources();
    blocksBuffer.releaseResources();
    surfaceBlockIndicesBuffer.releaseResources();
    surfaceBlockDispatch.releaseResources();
    surfaceHalfBlockDispatch.releaseResources();
    surfaceVerticesBuffer.releaseResources();
    surfaceVertexIndicesBuffer.releaseResources();
    surfaceVertDensityDispatch.releaseResources();
    surfaceVertDensityBuffer.releaseResources();
    surfaceVertexNormalBuffer.releaseResources();
}

void FluidScene::transitionBuffers(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState) {
    D3D12_RESOURCE_BARRIER cellParticleCountsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleCountBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER cellParticleIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellParticleIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER blocksBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        blocksBuffer.getBuffer(),
        beforeState,
        afterState
    );
    
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertexIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexIndicesBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertexNormalBufferBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertexNormalBuffer.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceHalfBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceHalfBlockDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER surfaceVertDensityDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVertDensityDispatch.getBuffer(),
        beforeState,
        afterState
    );

    D3D12_RESOURCE_BARRIER barriers[11] = { 
        cellParticleCountsBufferBarrier,
        cellParticleIndicesBufferBarrier,
        blocksBufferBarrier,
        surfaceBlockIndicesBufferBarrier,
        surfaceVerticesBufferBarrier,
        surfaceVertexIndicesBufferBarrier, 
        surfaceVertDensityBufferBarrier, 
        surfaceVertexNormalBufferBarrier2, 
        surfaceBlockDispatchBarrier, 
        surfaceHalfBlockDispatchBarrier, 
        surfaceVertDensityDispatchBarrier 
    };

    cmdList->ResourceBarrier(11, barriers);
}

void FluidScene::resetBuffers() {
	constexpr UINT THREAD_GROUP_SIZE = 256;
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numCellIndices = numCells * MAX_PARTICLES_PER_CELL;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);

	// Bind the PSO and Root Signature
    auto cmdList = bufferClearCP->getCommandList();
    cmdList->SetPipelineState(bufferClearCP->getPSO());
    cmdList->SetComputeRootSignature(bufferClearCP->getRootSignature());

    // Bind the descriptor heap
	ID3D12DescriptorHeap* descriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
	cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numCells, 0);
	cmdList->SetComputeRootDescriptorTable(1, cellParticleCountBuffer.getUAVGPUDescriptorHandle());
	cmdList->Dispatch((numCells + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numCellIndices, 0);
    cmdList->SetComputeRootDescriptorTable(1, cellParticleIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numCellIndices + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numBlocks, 0);
    cmdList->SetComputeRootDescriptorTable(1, blocksBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numBlocks + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVerticesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertexIndicesBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    cmdList->SetComputeRoot32BitConstants(0, 1, &numVerts, 0);
    cmdList->SetComputeRootDescriptorTable(1, surfaceVertDensityBuffer.getUAVGPUDescriptorHandle());
    cmdList->Dispatch((numVerts + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    context->executeCommandList(bufferClearCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);
    context->resetCommandList(bufferClearCP->getCommandListID());
}

/**
 * In one stream compaction step (detecting surface vertices), we end up with a total number (vertices),
 * which equals the number of the threads needed in subsequent compute steps (via indirect dispatch).
 * 
 * What we *need*, however, is the total number of thread groups, not threads. To avoid the latency of shuttling this data back and forth between the CPU and GPU,
 * we use a simple, one-thread compute pass to do the division, thus keeping the data on the GPU. To generalize this function for potential reuse, it accepts any dispatch buffer and any groupSize divisor.
 * 
 * This method assumes the dispatchArgs buffer is already a UAV.
 */
void FluidScene::divNumThreadsByGroupSize(StructuredBuffer* dispatchArgs, int groupSize) {
    auto cmdList = dispatchArgDivideCP->getCommandList();

    cmdList->SetPipelineState(dispatchArgDivideCP->getPSO());
    cmdList->SetComputeRootSignature(dispatchArgDivideCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    cmdList->SetComputeRootUnorderedAccessView(0, dispatchArgs->getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(1, 1, &groupSize, 0);

    cmdList->Dispatch(1, 1, 1);

    context->executeCommandList(dispatchArgDivideCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);
    context->resetCommandList(dispatchArgDivideCP->getCommandListID());
}