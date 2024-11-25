#include "Gram-SchmidtRootSignature.hlsl"

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
    float faceStrains[6];
    float shapeLambda[8];
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

    float compliance;
    float numSubsteps;
    uint partitionSize;
};

// Perform Gram-Schmidt orthonormalization to maintain voxel shape
void GramSchmidtOrthonormalization(inout float3 u, inout float3 v, inout float3 w) {
    float3 originalU = u;
    float3 originalV = v;
    float3 originalW = w;

    // Normalize first vector
    u = normalize(u);

    // Make v orthogonal to u and normalize
    v = v - dot(v, u) * u;
    if (length(v) < 1e-6) {
        v = cross(u, originalW);
        if (length(v) < 1e-6) {
            v = cross(u, float3(0, 1, 0));
        }
    }
    v = normalize(v);

    // Generate w as cross product to ensure right-handed system
    w = cross(u, v);
    w = normalize(w);
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= partitionSize) return;

    uint voxelIndex = partitionBuffer[DTid.x];
    Voxel voxel = voxels[voxelIndex];

    // Get current voxel axes
    float3 u = voxel.u;
    float3 v = voxel.v;
    float3 w = voxel.w;

    float h = deltaTime / numSubsteps;
    float alpha = compliance / (h * h);

    // Orthonormalize axes
    GramSchmidtOrthonormalization(u, v, w);

    // Update voxel axes
    voxel.u = u;
    voxel.v = v;
    voxel.w = w;

    // Calculate centroid
    float3 centroidPosition = float3(0, 0, 0);
    for (int i = 0; i < 8; ++i) {
        centroidPosition += particles[voxel.particleIndices[i]].position;
    }
    centroidPosition /= 8.0;

    // Define local positions for cubical voxel shape
    const float3 localPositions[8] = {
        float3(-0.05, -0.05, -0.05),
        float3(0.05, -0.05, -0.05),
        float3(0.05, 0.05, -0.05),
        float3(-0.05, 0.05, -0.05),
        float3(-0.05, -0.05, 0.05),
        float3(0.05, -0.05, 0.05),
        float3(0.05, 0.05, 0.05),
        float3(-0.05, 0.05, 0.05)
    };

    // Apply shape maintenance constraints
    [unroll]
        for (int j = 0; j < 8; ++j) {
            uint pIndex = voxel.particleIndices[j];
            Particle p = particles[pIndex];

            if (p.invMass > 0.0f) {
                float3 localPos = localPositions[j];
                // Transform local position to world space using orthonormal basis
                float3 worldPos = centroidPosition +
                    u * localPos.x +
                    v * localPos.y +
                    w * localPos.z;

                float3 constraint = p.position - worldPos;
                float w_inv = p.invMass;

                // XPBD update
                float C = dot(constraint, constraint);
                float lambda = voxel.shapeLambda[j];
                float deltaLambda = (-C - alpha * lambda) / (w_inv + alpha);
                voxel.shapeLambda[j] += deltaLambda;

                // Apply position correction
                p.position += constraint * w_inv * deltaLambda;
                particles[pIndex] = p;
            }
        }

    voxels[voxelIndex] = voxel;
}