#pragma once
#include <vector>
#include "Drawable.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"
#include "../D3D/StructuredBuffer.h"
#include "../Shaders/constants.h"
#include <Alembic/AbcCoreFactory/All.h>
#include <Alembic/AbcGeom/All.h>

using namespace Alembic::Abc;
using namespace Alembic::AbcGeom;

struct GridConstants {
    int numParticles;
    XMINT3 gridDim;
    XMFLOAT3 minBounds;
    float resolution;
};

// TODO: can just combine this with grid constants
struct MeshShadingConstants {
    XMMATRIX viewProj;
    XMINT3 dimensions;
    float resolution;
    XMFLOAT3 minBounds;
    float padding;
    XMFLOAT3 cameraPos;
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
               ComputePipeline* surfaceCellDetectionCP,
               ComputePipeline* surfaceVertexCompactionCP,
               ComputePipeline* surfaceVertexDensityCP,
               ComputePipeline* surfaceVertexNormalCP,
               MeshPipeline* fluidMeshPipeline);

    void compute(
        StructuredBuffer* positionsBuffer,
        int numParticles
    );
    void draw(Camera* camera);
    void constructScene();
    void computeBilevelUniformGrid();
    void computeSurfaceBlockDetection();
    void computeSurfaceCellDetection();
    void compactSurfaceVertices();
    void computeSurfaceVertexDensity();
    void computeSurfaceVertexNormal();
    void releaseResources();

private:
    void transitionBuffers(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);
    void resetBuffers(ID3D12GraphicsCommandList6* cmdList);

    GridConstants gridConstants;
    
    ComputePipeline* bilevelUniformGridCP;
    ComputePipeline* surfaceBlockDetectionCP;
    ComputePipeline* surfaceCellDetectionCP;
    ComputePipeline* surfaceVertexCompactionCP;
    ComputePipeline* surfaceVertexDensityCP;
    ComputePipeline* surfaceVertexNormalCP;

    MeshPipeline* fluidMeshPipeline;
    
    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;

	ID3D12CommandSignature* commandSignature = nullptr;
    ID3D12CommandSignature* meshCommandSignature = nullptr;

    StructuredBuffer* positionBuffer;
    StructuredBuffer cellParticleCountBuffer;
    StructuredBuffer cellParticleIndicesBuffer;
    StructuredBuffer blocksBuffer;
    StructuredBuffer surfaceBlockIndicesBuffer;
    StructuredBuffer surfaceBlockDispatch;
    StructuredBuffer surfaceHalfBlockDispatch; // This is just 2x surfaceBlockDispatch, but saves us a round trip to the GPU to multiply by 2
    StructuredBuffer surfaceVerticesBuffer;
    StructuredBuffer surfaceVertexIndicesBuffer;
    StructuredBuffer surfaceVertDensityDispatch;
    StructuredBuffer surfaceVertDensityBuffer;
    StructuredBuffer surfaceVertexNormalBuffer;

    StructuredBuffer blankCellParticleCountBuffer;
    StructuredBuffer blankCellParticleIndicesBuffer;
    StructuredBuffer blankBlocksBuffer;
    StructuredBuffer blankSurfaceVerticesBuffer;
    StructuredBuffer blankSurfaceVertDensityBuffer;
    StructuredBuffer blankSurfaceVertexNormalBuffer;
};