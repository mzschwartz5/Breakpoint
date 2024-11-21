#include "TestComputeRootSignature.hlsl"

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
};

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<Voxel> voxels : register(u1);
StructuredBuffer<uint> Indices : register(t0);

cbuffer ConstraintParams : register(b0) {
    uint constraintCount;
};

void GramSchmidtOrthonormalization(inout float3 u, inout float3 v, inout float3 w) {
    u = normalize(u);
    v = v - dot(v, u) * u;
    v = normalize(v);
    w = w - dot(w, u) * u - dot(w, v) * v;
    w = normalize(w);
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint voxelIndex = Indices[DTid.x];
    Voxel voxel = voxels[voxelIndex];

    // Perform Gram-Schmidt Orthonormalization
    float3 u = voxel.u;
    float3 v = voxel.v;
    float3 w = voxel.w;

    GramSchmidtOrthonormalization(u, v, w);

    // Update voxel axes
    voxel.u = u;
    voxel.v = v;
    voxel.w = w;

    // Calculate voxel centroid
    float3 centroidPosition = float3(0, 0, 0);
    for (int i = 0; i < 8; ++i) {
        centroidPosition += particles[voxel.particleIndices[i]].position;
    }
    centroidPosition /= 8.0;

    // Define local positions for a unit cube
    const float3 localPositions[8] = {
        float3(-0.05, -0.05, -0.05),
        float3(0.05, -0.05, -0.05),
        float3(0.05,  0.05, -0.05),
        float3(-0.05,  0.05, -0.05),
        float3(-0.05, -0.05,  0.05),
        float3(0.05, -0.05,  0.05),
        float3(0.05,  0.05,  0.05),
        float3(-0.05,  0.05,  0.05)
    };

    // Constraint parameters
    const float constraintStiffness = 1.0f;
    const float dt = 0.033f; // Timestep

    for (int j = 0; j < 8; ++j) {
        uint pIndex = voxel.particleIndices[j];
        Particle p = particles[pIndex];

        // Calculate world position based on local coordinates and orthonormal axes
        float3 localPos = localPositions[j];
        float3 worldPos = centroidPosition +
            u * localPos.x +
            v * localPos.y +
            w * localPos.z;

        // Store previous position before correction
        p.prevPosition = p.position;

        // Apply position correction with mass-based weighting and stiffness
        if (p.invMass > 0.0f) {
            float3 correction = worldPos - p.position;
            p.position += correction * p.invMass * constraintStiffness;

            // Update velocity based on position change
            p.velocity = (p.position - p.prevPosition) / dt;
        }

        particles[pIndex] = p;
    }

    // Write back updated voxel
    voxels[voxelIndex] = voxel;
}