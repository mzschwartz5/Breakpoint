#include "SurfaceVertexNormalsRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"

// Inputs
// SRV for the surface vertex density buffer
StructuredBuffer<float> surfaceVertexDensities : register(t0);
// SRV for the surface vertex indices
StructuredBuffer<int> surfaceVertexIndices : register(t1);
// Root SRV for dispatch params for this pass
StructuredBuffer<int3> surfaceVertDensityDispatch : register(t2);
// Root constants
ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

// Outputs
// UAV for the surface vertex normals
RWStructuredBuffer<float3> surfaceVertexNormals : register(u0);

[numthreads(SURFACE_VERTEX_NORMAL_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= (surfaceVertDensityDispatch[0].x * SURFACE_VERTEX_DENSITY_THREADS_X)) { // TODO: this might overshoot. Need total number of surface verts.
        return;
    }

    int globalSurfaceVertIndex1d = surfaceVertexIndices[globalThreadId.x]; // I wonder if you could encode the 3D index into one entry in the buffer, to avoid the modular arithmetic in to3D
    int3 globalSurfaceVertIndex3d = to3D(globalSurfaceVertIndex1d, cb.dimensions + int3(1, 1, 1)); 

    float3 normal;
    normal.x = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + int3(1, 0, 0), cb.dimensions + int3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - int3(1, 0, 0), cb.dimensions + int3(1, 1, 1))];

    normal.y = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + int3(0, 1, 0), cb.dimensions + int3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - int3(0, 1, 0), cb.dimensions + int3(1, 1, 1))];
    
    normal.z = surfaceVertexDensities[to1D(globalSurfaceVertIndex3d + int3(0, 0, 1), cb.dimensions + int3(1, 1, 1))]
                - surfaceVertexDensities[to1D(globalSurfaceVertIndex3d - int3(0, 0, 1), cb.dimensions + int3(1, 1, 1))];

    normal /= cb.resolution; // Even though this doesn't affect the direction of the normal, it's necessary for numerical stability, when normal components are close to zero.
    normal = normalize(normal);

    // Like densities buffer, not compressed, but the only populated entries are for surface vertices.
    surfaceVertexNormals[globalSurfaceVertIndex1d] = normal;
}