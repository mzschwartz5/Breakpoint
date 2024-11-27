#include "Gram-SchmidtRootSignature.hlsl"

struct Particle {
    float3 position;
    float3 prevPosition;
    float3 velocity;
    float invMass;
};

struct Voxel {
    uint particleIndices[8];
    float3 u;
    float3 v;
    float3 w;
    bool faceConnections[6];
    float faceStrains[6];
    float faceLambdas[6];
};

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<Voxel> voxels : register(u1);
StructuredBuffer<uint> partitionBuffer : register(t0);

cbuffer SimulationParams : register(b0) {
    uint constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float3 gravity;
    float numSubsteps;
    float compliance;
    uint partitionSize;
};

float Random(float2 seed) {
    float dotProduct = dot(seed, float2(12.9898, 78.233));
    float sinValue = sin(dotProduct) * 43758.5453;
    return frac(sinValue);
}

float CalculateFaceStrain(float3 normal, float3 expectedNormal) {
    return length(normal - expectedNormal);
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= partitionSize) return;

    uint voxelIndex = partitionBuffer[DTid.x];
    Voxel voxel = voxels[voxelIndex];

    float h = deltaTime / numSubsteps;
    float alpha = compliance / (h * h);

    // Calculate voxel centroid
    float3 centroidPosition = float3(0, 0, 0);
    [unroll]
        for (int i = 0; i < 8; ++i) {
            centroidPosition += particles[voxel.particleIndices[i]].position;
        }
    centroidPosition /= 8.0;

    // Face vertex indices for each face
    const uint faceIndices[6][4] = {
        {1, 2, 6, 5}, // +X face
        {0, 3, 7, 4}, // -X face
        {2, 3, 7, 6}, // +Y face
        {0, 1, 5, 4}, // -Y face
        {4, 5, 6, 7}, // +Z face
        {0, 1, 2, 3}  // -Z face
    };

    // Process each face
    [unroll]
        for (int face = 0; face < 6; face++) {
            if (voxel.faceConnections[face]) {
                float totalInvMass = 0;

                // Calculate face center
                float3 faceCenter = float3(0, 0, 0);
                for (int fi = 0; fi < 4; fi++) {
                    uint pIdx = voxel.particleIndices[faceIndices[face][fi]];
                    faceCenter += particles[pIdx].position;
                    totalInvMass += particles[pIdx].invMass;
                }
                faceCenter /= 4.0;

                // Determine expected normal based on local axes
                float3 expectedNormal;
                if (face < 2) expectedNormal = (face == 0) ? voxel.u : -voxel.u;
                else if (face < 4) expectedNormal = (face == 2) ? voxel.v : -voxel.v;
                else expectedNormal = (face == 4) ? voxel.w : -voxel.w;

                // Calculate actual face normal
                float3 actualNormal = normalize(faceCenter - centroidPosition);

                // Calculate strain and check for breaking
                float strain = CalculateFaceStrain(actualNormal, expectedNormal);
                voxel.faceStrains[face] = strain;

                float lambda = voxel.faceLambdas[face];
                float deltaLambda = (-strain - alpha * lambda) / (totalInvMass + alpha);
                voxel.faceLambdas[face] += deltaLambda;

                // Check for breaking with random variation
                float randomVariation = Random(float2(voxelIndex + face, randomSeed));
                float adjustedThreshold = breakingThreshold * (0.8 + 0.4 * randomVariation);

                if (strain > adjustedThreshold) {
                    voxel.faceConnections[face] = false;
                    voxel.faceLambdas[face] = 0.0f;
                }
                else if (voxel.faceConnections[face]) {
                    float3 correction = (expectedNormal - actualNormal) * deltaLambda;

                    // Apply corrections to face vertices
                    for (int fi = 0; fi < 4; fi++) {
                        uint pIdx = voxel.particleIndices[faceIndices[face][fi]];
                        Particle p = particles[pIdx];

                        if (p.invMass > 0.0f) {
                            float3 toParticle = p.position - centroidPosition;
                            float weight = length(toParticle) / length(faceCenter - centroidPosition);

                            p.position += correction * p.invMass * weight;
                            particles[pIdx] = p;
                        }
                    }
                }
            }
        }

    voxels[voxelIndex] = voxel;
}