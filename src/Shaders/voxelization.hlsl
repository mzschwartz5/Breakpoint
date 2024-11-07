#define VOXELIZATION_ROOTSIG \
"DescriptorTable(SRV(t0, numDescriptors=2))," \ // Vertex and Index buffers
"DescriptorTable(UAV(u0)),"                    \ // Voxel grid UAV
"RootConstants(num32BitConstants=7, b0)"         // Voxelization parameters

#include "RootSignature.hlsl"

cbuffer VoxelizationConstants : register(b0) {
    float4x4 WorldViewProjection;
    uint voxelGridSize;
    float voxelSize;
    float3 voxelGridMin; // Minimum bounds of the voxel grid
}

[numthreads(8, 8, 8)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint3 voxelPos = DTid;

    if (voxelPos.x >= voxelGridSize || voxelPos.y >= voxelGridSize || voxelPos.z >= voxelGridSize)
        return;

    // Compute world position of the voxel center
    float3 worldPos = voxelGridMin + (float3(voxelPos) + 0.5f) * voxelSize;

    // For simplicity, assuming a single triangle
    float3 v0 = vertices[indices[0]];
    float3 v1 = vertices[indices[1]];
    float3 v2 = vertices[indices[2]];

    // Simple bounding box overlap test
    float halfSize = voxelSize * 0.5f;
    float3 voxelMin = worldPos - halfSize;
    float3 voxelMax = worldPos + halfSize;

    float3 triMin = min(v0, min(v1, v2));
    float3 triMax = max(v0, max(v1, v2));

    bool overlaps = (voxelMin.x <= triMax.x && voxelMax.x >= triMin.x) &&
        (voxelMin.y <= triMax.y && voxelMax.y >= triMin.y) &&
        (voxelMin.z <= triMax.z && voxelMax.z >= triMin.z);

    if (overlaps) {
        voxelGrid[voxelPos] = 1;
    }
}

