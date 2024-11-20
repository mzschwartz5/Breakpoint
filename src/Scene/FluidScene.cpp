#include "FluidScene.h"

FluidScene::FluidScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* bilevelUniformGridCP, ComputePipeline* surfaceBlockDetectionCP)
    : Drawable(context, pipeline), bilevelUniformGridCP(bilevelUniformGridCP), surfaceBlockDetectionCP(surfaceBlockDetectionCP)
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
    
    std::vector<Cell> cells(numCells);
    std::vector<Block> blocks(numBlocks);
    std::vector<unsigned int> surfaceBlockIndices(numBlocks, 0);
    std::vector<unsigned int> surfaceBlockCountCPU(1, 0);

    blocksBuffer = StructuredBuffer(blocks.data(), numBlocks, sizeof(Block), bilevelUniformGridCP->getDescriptorHeap());
    blocksBuffer.passUAVDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    cellsBuffer = StructuredBuffer(cells.data(), numCells, sizeof(Cell), bilevelUniformGridCP->getDescriptorHeap());
    cellsBuffer.passUAVDataToGPU(*context, bilevelUniformGridCP->getCommandList(), bilevelUniformGridCP->getCommandListID());

    surfaceBlockIndicesBuffer = StructuredBuffer(surfaceBlockIndices.data(), numBlocks, sizeof(unsigned int), surfaceBlockDetectionCP->getDescriptorHeap());
    surfaceBlockIndicesBuffer.passUAVDataToGPU(*context, surfaceBlockDetectionCP->getCommandList(), surfaceBlockDetectionCP->getCommandListID());

    surfaceBlockCount = StructuredBuffer(surfaceBlockCountCPU.data(), 1, sizeof(unsigned int), surfaceBlockDetectionCP->getDescriptorHeap());
    surfaceBlockCount.passUAVDataToGPU(*context, surfaceBlockDetectionCP->getCommandList(), surfaceBlockDetectionCP->getCommandListID());

    // Create fence
    context->getDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
}

void FluidScene::compute() {
    // TODO: do a reset compute pass first to clear the buffers (take advantage of existing passes where possible)
    // (This is a todo, because I need to implement a third compute pass that operates on the cell level to clear the cell buffers)

    // ======= Bilevel Uniform Grid Compute Pipeline =======
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
    int workgroupSize = (gridConstants.numParticles + BILEVEL_UNIFORM_GRID_THREADS_X - 1) / BILEVEL_UNIFORM_GRID_THREADS_X;
    cmdList->Dispatch(workgroupSize, 1, 1);

    // Transition blocksBuffer from UAV to SRV for the next pass
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = blocksBuffer.getBuffer();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(1, &barrier);

    // Execute command list
    context->executeCommandList(bilevelUniformGridCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    // Reinitialize command list
    context->resetCommandList(bilevelUniformGridCP->getCommandListID());

    // ======= Surface Block Detection Compute Pipeline =======
    cmdList = surfaceBlockDetectionCP->getCommandList();

    cmdList->SetPipelineState(surfaceBlockDetectionCP->getPSO());
    cmdList->SetComputeRootSignature(surfaceBlockDetectionCP->getRootSignature());

    // Set descriptor heap
    ID3D12DescriptorHeap* computeDescriptorHeaps2[] = { surfaceBlockDetectionCP->getDescriptorHeap()->Get() };
    cmdList->SetDescriptorHeaps(_countof(computeDescriptorHeaps2), computeDescriptorHeaps2);

    int numCells = gridConstants.gridDim.x * gridConstants.gridDim.y * gridConstants.gridDim.z;
    int numBlocks = numCells / (CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE);

    // Set compute root descriptor table
    cmdList->SetComputeRootDescriptorTable(0, blocksBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootDescriptorTable(1, surfaceBlockIndicesBuffer.getGPUDescriptorHandle());
    cmdList->SetComputeRootUnorderedAccessView(2, surfaceBlockCount.getGPUVirtualAddress());
    cmdList->SetComputeRoot32BitConstants(3, 1, &numBlocks, 0);

    // Dispatch
    workgroupSize = (numBlocks + SURFACE_BLOCK_DETECTION_THREADS_X - 1) / SURFACE_BLOCK_DETECTION_THREADS_X;
    cmdList->Dispatch(workgroupSize, 1, 1);

    // Transition blocksBuffer back to UAV for the next frame
    barrier.Transition.pResource = blocksBuffer.getBuffer();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    
    cmdList->ResourceBarrier(1, &barrier);

    // Execute command list
    context->executeCommandList(surfaceBlockDetectionCP->getCommandListID());
    context->signalAndWaitForFence(fence, fenceValue);

    // Reinitialize command list
    context->resetCommandList(surfaceBlockDetectionCP->getCommandListID());

    // Copy surface block count to CPU
    int surfaceBlockCountCPU;
    surfaceBlockCount.copyDataFromGPU(*context, &surfaceBlockCountCPU, surfaceBlockDetectionCP->getCommandList(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, surfaceBlockDetectionCP->getCommandListID());
}

void FluidScene::releaseResources() {
    renderPipeline->releaseResources();
}