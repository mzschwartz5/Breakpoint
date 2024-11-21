#include "FluidScene.h"

FluidScene::FluidScene(DXContext* context, 
                       RenderPipeline* pipeline, 
                       ComputePipeline* bilevelUniformGridCP, 
                       ComputePipeline* surfaceBlockDetectionCP,
                       ComputePipeline* surfaceCellDetectionCP)
    : Drawable(context, pipeline), 
      bilevelUniformGridCP(bilevelUniformGridCP), 
      surfaceBlockDetectionCP(surfaceBlockDetectionCP),
      surfaceCellDetectionCP(surfaceCellDetectionCP)
{
    constructScene();
}

void FluidScene::draw(Camera* camera) {

}

float getRandomFloatInRange(float min, float max) {
    return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
}

void FluidScene::constructScene() {
    unsigned int numParticles = 3 * CELLS_PER_BLOCK_EDGE * 3 * CELLS_PER_BLOCK_EDGE * 3 * CELLS_PER_BLOCK_EDGE;
    gridConstants = { numParticles, {3 * CELLS_PER_BLOCK_EDGE, 3 * CELLS_PER_BLOCK_EDGE, 3 * CELLS_PER_BLOCK_EDGE}, {0.f, 0.f, 0.f}, 0.1f };

    // Populate position data. 1000 partices in a 12x12x12 block of cells, each at a random position in a cell.
    // (Temporary, eventually, position data will come from simulation)
    for (int i = 0; i < gridConstants.gridDim.x; ++i) {
        for (int j = 0; j < gridConstants.gridDim.y; ++j) {
            for (int k = 0; k < gridConstants.gridDim.z; ++k) {
                positions.push_back({ 
                    gridConstants.minBounds.x + gridConstants.resolution * i + getRandomFloatInRange(0.f, gridConstants.resolution),
                    gridConstants.minBounds.y + gridConstants.resolution * j + getRandomFloatInRange(0.f, gridConstants.resolution),
                    gridConstants.minBounds.z + gridConstants.resolution * k + getRandomFloatInRange(0.f, gridConstants.resolution) 
                });
            }
        }
    }

    positionBuffer = StructuredBuffer(positions.data(), gridConstants.numParticles, sizeof(XMFLOAT3), bilevelUniformGridCP->getDescriptorHeap());
    positionBuffer.passSRVDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    // Create cells and blocks buffers
    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);
    int numVerts = (gridConstants.gridDim.x + 1) * (gridConstants.gridDim.y + 1) * (gridConstants.gridDim.z + 1);
    
    std::vector<Cell> cells(numCells);
    std::vector<Block> blocks(numBlocks);
    std::vector<unsigned int> surfaceBlockIndices(numBlocks, 0);
    XMUINT3 surfaceBlockDispatchCPU = { 0, 0, 0 };
    std::vector<unsigned int> surfaceVertices(numVerts, 0);

    blocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block), bilevelUniformGridCP->getDescriptorHeap());
    blocksBuffer.passUAVDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    cellsBuffer = StructuredBuffer(cells.data(), numCells, sizeof(Cell), bilevelUniformGridCP->getDescriptorHeap());
    cellsBuffer.passUAVDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    surfaceBlockIndicesBuffer = StructuredBuffer(surfaceBlockIndices.data(), numBlocks, sizeof(unsigned int), surfaceBlockDetectionCP->getDescriptorHeap());
    surfaceBlockIndicesBuffer.passUAVDataToGPU(*context, surfaceBlockDetectionCP->getCommandList(), surfaceBlockDetectionCP->getCommandListID());

    surfaceBlockDispatch = StructuredBuffer(&surfaceBlockDispatchCPU, 1, sizeof(XMUINT3), surfaceBlockDetectionCP->getDescriptorHeap());
    surfaceBlockDispatch.passUAVDataToGPU(*context, surfaceBlockDetectionCP->getCommandList(), surfaceBlockDetectionCP->getCommandListID());

    surfaceVerticesBuffer = StructuredBuffer(surfaceVertices.data(), numVerts, sizeof(unsigned int), surfaceCellDetectionCP->getDescriptorHeap());
    surfaceVerticesBuffer.passUAVDataToGPU(*context, surfaceCellDetectionCP->getCommandList(), surfaceCellDetectionCP->getCommandListID());

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

    // Create fence
    context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
}

void FluidScene::compute() {
    // TODO: do a reset compute pass first to clear the buffers (take advantage of existing passes where possible)
    // (This is a todo, because I need to implement a third compute pass that operates on the cell level to clear the cell buffers)
    computeBilevelUniformGrid();
    computeSurfaceBlockDetection();
    computeSurfaceCellDetection();
}

void FluidScene::computeBilevelUniformGrid() {
    auto cmdList = bilevelUniformGridCP->getCommandList();

    cmdList->SetPipelineState(bilevelUniformGridCP->getPSO());
    cmdList->SetComputeRootSignature(bilevelUniformGridCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { bilevelUniformGridCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, positionBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellsBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, blocksBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRoot32BitConstants(3, 8, &gridConstants, 0);

    // Dispatch
    int numWorkGroups = (gridConstants.numParticles + BILEVEL_UNIFORM_GRID_THREADS_X - 1) / BILEVEL_UNIFORM_GRID_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);

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
}

void FluidScene::computeSurfaceBlockDetection() {
    auto cmdList = surfaceBlockDetectionCP->getCommandList();

    cmdList->SetPipelineState(surfaceBlockDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceBlockDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { surfaceBlockDetectionCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, blocksBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceBlockIndicesBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 1, &numBlocks, 0);

    // Dispatch
    int numWorkGroups = (numBlocks + SURFACE_BLOCK_DETECTION_THREADS_X - 1) / SURFACE_BLOCK_DETECTION_THREADS_X;
    cmdList->Dispatch(numWorkGroups, 1, 1);

    // Transition blocksBuffer back to UAV for the next frame
    D3D12_RESOURCE_BARRIER blocksBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        blocksBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    // Transition surfaceBlockIndicesBuffer to UAV for the next pass
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    // Transition surfaceBlockDispatch to an SRV
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition cellsBuffer to SRV for the next pass
    D3D12_RESOURCE_BARRIER cellsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellsBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    D3D12_RESOURCE_BARRIER barriers[4] = { blocksBufferBarrier, surfaceBlockIndicesBufferBarrier, surfaceBlockDispatchBarrier, cellsBufferBarrier };
    cmdList->ResourceBarrier(1, barriers);

    context->executeCommandList(surfaceBlockDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceBlockDetectionCP->getCommandListID());
}

void FluidScene::computeSurfaceCellDetection() {
    auto cmdList = surfaceCellDetectionCP->getCommandList();

    cmdList->SetPipelineState(surfaceCellDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceCellDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps[] = { surfaceCellDetectionCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps), computeDescriptorHeaps);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, surfaceBlockIndicesBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, cellsBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(2, surfaceVerticesBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootShaderResourceView(3, surfaceBlockDispatch.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(4, 3, &gridConstants.gridDim, 0);

    // Transition surfaceBlockDispatch to indirect argument buffer
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
    );

    cmdList->ResourceBarrier(1, &surfaceBlockDispatchBarrier);

    // Dispatch
    cmdList->ExecuteIndirect(commandSignature, 1, surfaceBlockDispatch.getBuffer(), 0, nullptr, 0);

    // Transition surfaceBlockDispatch back to UAV for the next frame
    D3D12_RESOURCE_BARRIER surfaceBlockDispatchBarrier2 = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockDispatch.getBuffer(),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition surfaceBlockIndicesBuffer back to UAV for the next frame
    D3D12_RESOURCE_BARRIER surfaceBlockIndicesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceBlockIndicesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    // Transition cellsBuffer back to UAV for the next frame
    D3D12_RESOURCE_BARRIER cellsBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        cellsBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    // Transition surfaceVerticesBuffer back to SRV for the next pass
    D3D12_RESOURCE_BARRIER surfaceVerticesBufferBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
        surfaceVerticesBuffer.getBuffer(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE 
    );

    D3D12_RESOURCE_BARRIER barriers[4] = { surfaceBlockDispatchBarrier2, surfaceBlockIndicesBufferBarrier, cellsBufferBarrier, surfaceVerticesBufferBarrier };
    cmdList->ResourceBarrier(1, barriers);

    context->executeCommandList(surfaceCellDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    context->resetCommandList(surfaceCellDetectionCP->getCommandListID());
}

void FluidScene::releaseResources() {
    renderPipeline->releaseResources();
    bilevelUniformGridCP->releaseResources();
    surfaceBlockDetectionCP->releaseResources();
    surfaceCellDetectionCP->releaseResources();
    positionBuffer.releaseResources();
    cellsBuffer.releaseResources();
    blocksBuffer.releaseResources();
    surfaceBlockIndicesBuffer.releaseResources();
    surfaceBlockDispatch.releaseResources();
    surfaceVerticesBuffer.releaseResources();
}