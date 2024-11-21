#pragma once
#include <vector>
#include "Drawable.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/StructuredBuffer.h"
#include "../Shaders/constants.h"

struct GridConstants {
    unsigned int numParticles;
    XMINT3 gridDim;
    XMFLOAT3 minBounds;
    float resolution;
};

struct Cell {
    int particleCount;
    int particleIndices[MAX_PARTICLES_PER_CELL];
};

struct Block {
    int nonEmptyCellCount;
};

class FluidScene : public Drawable {
public:
    FluidScene() = delete;
    FluidScene(DXContext* context, 
               RenderPipeline *pipeline, 
               ComputePipeline* bilevelUniformGridCP, 
               ComputePipeline* surfaceBlockDetectionCP,
               ComputePipeline* surfaceCellDetectionCP);

    void compute();
    void draw(Camera* camera);
    void constructScene();
    void computeBilevelUniformGrid();
    void computeSurfaceBlockDetection();
    void computeSurfaceCellDetection();
    void releaseResources();

private:
    GridConstants gridConstants;
    
    ComputePipeline* bilevelUniformGridCP;
    ComputePipeline* surfaceBlockDetectionCP;
    ComputePipeline* surfaceCellDetectionCP;
    
    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;

	ID3D12CommandSignature* commandSignature = nullptr;

    std::vector<XMFLOAT3> positions;
	StructuredBuffer positionBuffer;
    StructuredBuffer cellsBuffer;
    StructuredBuffer blocksBuffer;
    StructuredBuffer surfaceBlockIndicesBuffer;
    StructuredBuffer surfaceBlockDispatch;
    StructuredBuffer surfaceVerticesBuffer;
};