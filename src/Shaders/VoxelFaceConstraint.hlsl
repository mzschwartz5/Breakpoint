// Root Signature
// "DescriptorTable(UAV(u0)), DescriptorTable(SRV(t0)), RootConstants(num32BitConstants=1, b0)"

#include "RootSignature.hlsl"

struct Voxel {
    float3 u;
    float3 v;
    float3 w;
};

struct FaceConstraint {
    uint voxelA;     
    uint voxelB;     
    uint faceIndex;  // Index representing which face is shared (0 for u, 1 for v, 2 for w)
};

RWStructuredBuffer<Voxel> voxels : register(u0);
StructuredBuffer<FaceConstraint> faceConstraints : register(t0);

[RootSignature(ROOTSIG)]
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint constraintIndex = DTid.x;
    FaceConstraint fc = faceConstraints[constraintIndex];

    Voxel voxelA = voxels[fc.voxelA];
    Voxel voxelB = voxels[fc.voxelB];

    //shared axes
    switch (fc.faceIndex) {
    case 0: // Face aligned along u-axis
        float3 avgU = normalize((voxelA.u + voxelB.u) * 0.5);
        voxelA.u = avgU;
        voxelB.u = avgU;
        break;
    case 1: // Face aligned along v-axis
        float3 avgV = normalize((voxelA.v + voxelB.v) * 0.5);
        voxelA.v = avgV;
        voxelB.v = avgV;
        break;
    case 2: // Face aligned along w-axis
        float3 avgW = normalize((voxelA.w + voxelB.w) * 0.5);
        voxelA.w = avgW;
        voxelB.w = avgW;
        break;
    }

    voxels[fc.voxelA] = voxelA;
    voxels[fc.voxelB] = voxelB;
}
