#pragma once
#include <vector>
#include "Drawable.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/StructuredBuffer.h"

struct GridConstants {
    XMINT3 gridDim;
    XMFLOAT3 minBounds;
    float resolution;
    unsigned int numParticles;
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