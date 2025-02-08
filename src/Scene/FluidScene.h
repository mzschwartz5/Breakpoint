#pragma once
#include <vector>
#include "Drawable.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"
#include "../D3D/StructuredBuffer.h"
#include "../Shaders/constants.h"
#include "../Support/AlembicLoader.h"

struct GridConstants {
    int numParticles;
    XMINT3 gridDim; // length / width / height in # of grid cells
    XMFLOAT3 minBounds; // origin of the grid area
    float resolution; // size of each grid cell
};

// TODO: can just combine this with grid constants
struct MeshShadingConstants {
    XMMATRIX viewProj;
    XMINT3 dimensions;
    float resolution;
    XMFLOAT3 minBounds;
    int screenWidth; // split up into two floats to match the 16 byte alignment
    XMFLOAT3 cameraPos;
    int screenHeight;  // split up into two floats to match the 16 byte alignment
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
               ComputePipeline* bufferClearCP,
               ComputePipeline* dispatchArgDivideCP,
               MeshPipeline* fluidMeshPipeline);

    void compute();
    void draw(
        Camera* camera,
        D3D12_GPU_DESCRIPTOR_HANDLE objectColorTextureHandle,
        D3D12_GPU_DESCRIPTOR_HANDLE objectPositionTextureHandle,
        int screenWidth,
        int screenHeight
    );
    void constructScene();
    void computeBilevelUniformGrid();
    void computeSurfaceBlockDetection();
    void computeSurfaceCellDetection();
    void compactSurfaceVertices();
    void computeSurfaceVertexDensity();
    void computeSurfaceVertexNormal();
    void releaseResources();
    DescriptorHeap* getSRVHeapForRenderTextures() {
        return bilevelUniformGridCP->getDescriptorHeap();
    }

private:
    void transitionBuffers(ID3D12GraphicsCommandList6* cmdList, D3D12_RESOURCE_STATES beforeState, D3D12_RESOURCE_STATES afterState);
    void resetBuffers();
    void divNumThreadsByGroupSize(StructuredBuffer* divNumThreadsByGroupSize, int groupSize);

    GridConstants gridConstants;
    
    ComputePipeline* bilevelUniformGridCP;
    ComputePipeline* surfaceBlockDetectionCP;
    ComputePipeline* surfaceCellDetectionCP;
    ComputePipeline* surfaceVertexCompactionCP;
    ComputePipeline* surfaceVertexDensityCP;
    ComputePipeline* surfaceVertexNormalCP;
    ComputePipeline* bufferClearCP;
    ComputePipeline* dispatchArgDivideCP;

    MeshPipeline* fluidMeshPipeline;
    
    UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;

	ID3D12CommandSignature* commandSignature = nullptr;
    ID3D12CommandSignature* meshCommandSignature = nullptr;

    StructuredBuffer positionBuffer;
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
};