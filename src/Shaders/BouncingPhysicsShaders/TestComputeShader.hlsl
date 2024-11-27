#include "TestComputeRootSignature.hlsl"  // Includes the ROOTSIG definition


struct Particle {
    float3 position;
    float3 prevPosition;
    float3 velocity;
    float invMass;
};

struct Voxel {
    uint particleIndices[8];
    float3 u; // Local X-axis
    float3 v; // Local Y-axis
    float3 w; // Local Z-axis
    bool faceConnections[6]; // Store connection state for each face (+X,-X,+Y,-Y,+Z,-Z)
    float faceStrains[6]; // Store strain for each face

    float shapeLambda[8];
};


RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<Voxel> voxels : register(u1);
StructuredBuffer<uint> shapepartitionBuffer : register(t0);
StructuredBuffer<uint> xpepartitionBuffer : register(t1);
StructuredBuffer<uint> ypartitionBuffer : register(t2);
StructuredBuffer<uint> zpartitionBuffer : register(t3);

cbuffer SimulationParams : register(b0) {
    uint constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float3 gravity;

    float compliance;
    float numSubsteps;
    uint partitionSize;
};
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    

}