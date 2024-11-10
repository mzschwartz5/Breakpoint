#define GRAM_SCHMIDT_ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"DescriptorTable(UAV(u0, numDescriptors = 2))," \
"RootConstants(num32BitConstants=32, b0)," \
"DescriptorTable(SRV(t0))"

#include "RootSignature.hlsl"




struct Particle {
    float3 position;
    float inverseMass;
};

struct Voxel {
    uint particleIndices[8]; 
    float3 u; // Local X-axis
    float3 v; // Local Y-axis
    float3 w; // Local Z-axis
};

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<Voxel> voxels : register(u1);
StructuredBuffer<uint> Indices : register(t0);

[RootSignature(ROOTSIG)]
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint voxelIndex = Indices[DTid.x];
    Voxel voxel = voxels[voxelIndex];

    float3 u = voxel.u;
    float3 v = voxel.v;
    float3 w = voxel.w;

    // Gram-Schmidt Orthonormalization
    u = normalize(u);
    v = v - dot(v, u) * u;
    v = normalize(v);
    w = w - dot(w, u) * u - dot(w, v) * v;
    w = normalize(w);

    voxel.u = u;
    voxel.v = v;
    voxel.w = w;

    // Update particle positions based on the new axes
    float3 centroidPosition = float3(0, 0, 0);
    for (int i = 0; i < 8; ++i) {
        centroidPosition += + particles[voxel.particleIndices[i]].position;
    }
    centroidPosition /= 8.0;

    //define particles local position
    float3 localPositions[8] = {
        float3(-0.5, -0.5, -0.5),
        float3(0.5, -0.5, -0.5),
        float3(0.5,  0.5, -0.5),
        float3(-0.5,  0.5, -0.5),
        float3(-0.5, -0.5,  0.5),
        float3(0.5, -0.5,  0.5),
        float3(0.5,  0.5,  0.5),
        float3(-0.5,  0.5,  0.5)
    };

    for (int j = 0; j < 8; ++j) {
        uint pIndex = voxel.particleIndices[j];
        Particle p = particles[pIndex];

        float3 localPos = localPositions[j];
        float3 worldPos = centroidPosition + u * localPos.x + v * localPos.y + w * localPos.z;

        // Apply position correction
        float3 correction = worldPos - p.position;
        if (p.inverseMass > 0.0f) {
            p.position += correction * p.inverseMass;
        }

        particles[pIndex] = p;
    }

    // Write back
    voxels[voxelIndex] = voxel;
}
