#include "SurfaceVertexNormalsRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"

// Inputs
// SRV for the surface vertex density buffer
StructuredBuffer<float> surfaceVertexDensities : register(t0);
// SRV for the surface vertex indices
StructuredBuffer<uint> surfaceVertexIndices : register(t1);
// Root SRV for dispatch params for this pass
StructuredBuffer<uint3> surfaceVertDensityDispatch : register(t2);
// Root constants
ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

// Outputs
// UAV for the surface vertex normals
// Note: to save on memory bandwidth, store normals as float2s (with z component implied by normalization constraint). 
// TODO: consider compressing further using packed normals. (Also pack particle positions in other compute passes?)
RWStructuredBuffer<float2> surfaceVertexNormals : register(u0);


// TODO: 3D dispatch for better indexing
[numthreads(SURFACE_VERTEX_NORMAL_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= surfaceVertDensityDispatch[0].x) {
        return;
    }

    uint globalSurfaceVertIndex1d = surfaceVertexIndices[globalThreadId.x];
    uint3 globalSurfaceVertIndex3d = to3D(globalSurfaceVertIndex1d, (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1));

    float3 normal;
    normal.x = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + uint3(1, 0, 0), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - uint3(1, 0, 0), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))];

    normal.y = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + uint3(0, 1, 0), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - uint3(0, 1, 0), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))];
    
    normal.z = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + uint3(0, 0, 1), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - uint3(0, 0, 1), (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1))];

    normal /= cb.resolution;
    normal = normalize(normal);

    surfaceVertexNormals[globalSurfaceVertIndex1d] = float2(normal.x, normal.y);
}