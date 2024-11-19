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
    FluidScene(DXContext* context, RenderPipeline *pipeline, ComputePipeline* bilevelUniformGridCP);

    void compute();
    void draw(Camera* camera);
    void constructScene();
    void releaseResources();

private:
    GridConstants gridConstants;
    
    ComputePipeline* bilevelUniformGridCP;
    std::vector<XMFLOAT3> positions;
	StructuredBuffer positionBuffer;
    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;

    StructuredBuffer cellsBuffer;
    StructuredBuffer blocksBuffer;
};