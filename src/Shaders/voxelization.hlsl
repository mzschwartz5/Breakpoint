#include "RootSignature.hlsl"

StructuredBuffer<float3> vertices : register(t0);
StructuredBuffer<uint> indices : register(t1);
AppendStructuredBuffer<float4x4> outputModelMatrices : register(u0);

cbuffer VoxelizationParams : register(b0) {
    uint voxelGridSize;
    float voxelSize;
    float3 voxelGridMin;
}
bool TriangleIntersectsVoxel(float3 v0, float3 v1, float3 v2, float3 voxelCenter, float voxelSize) {
    // triangle-voxel intersection test
    //simple axis-aligned bounding box overlap test
    float halfSize = voxelSize * 0.5f;
    float3 voxelMin = voxelCenter - halfSize;
    float3 voxelMax = voxelCenter + halfSize;

    // Compute bounding box
    float3 triMin = min(v0, min(v1, v2));
    float3 triMax = max(v0, max(v1, v2));

    // Check for overlap
    return (voxelMin.x <= triMax.x && voxelMax.x >= triMin.x) &&
        (voxelMin.y <= triMax.y && voxelMax.y >= triMin.y) &&
        (voxelMin.z <= triMax.z && voxelMax.z >= triMin.z);
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint3 voxelPos = DTid;

    if (voxelPos.x >= voxelGridSize || voxelPos.y >= voxelGridSize || voxelPos.z >= voxelGridSize)
        return;

    // Compute world position of the voxel center
    float3 voxelCenter = voxelGridMin + (float3(voxelPos) + 0.5f) * voxelSize;

   
   
   
    bool isOccupied = false;

    for (uint i = 0; i < indices.Length / 3; ++i) {
        // Get triangle indices
        uint index0 = indices[i * 3 + 0];
        uint index1 = indices[i * 3 + 1];
        uint index2 = indices[i * 3 + 2];

        // Get triangle vertices
        float3 v0 = vertices[index0];
        float3 v1 = vertices[index1];
        float3 v2 = vertices[index2];

        // Perform triangle-voxel intersection test
        if (TriangleIntersectsVoxel(v0, v1, v2, voxelCenter, voxelSize)) {
            isOccupied = true;
            break;
        }
    }

    if (isOccupied) {
        // Create model matrix for the voxel 
        float4x4 modelMatrix = float4x4(
            voxelSize, 0, 0, 0,
            0, voxelSize, 0, 0,
            0, 0, voxelSize, 0,
            voxelCenter.x, voxelCenter.y, voxelCenter.z, 1
        );

        // Append the model matrix to the output buffer
        outputModelMatrices.Append(modelMatrix);
    }
}



