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
    float faceStrains[6]; // Store strain for each face


    float3 centroidVelocity; 
    float accumulatedStrain;
};


RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<Voxel> voxels : register(u1);
StructuredBuffer<uint> Indices : register(t0);

cbuffer SimulationParams : register(b0) {
    uint constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float3 gravity;

    float strainMemory; 
    float rotationalInertia;
};


float Random(float2 seed) {
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
}

float CalculateFaceStrain(float3 normal, float3 expectedNormal, float3 shearDir) {
    float normalStrain = length(normal - expectedNormal);
    float shearStrain = abs(dot(normal, shearDir));
    return normalStrain + shearStrain * 0.5; // Weight shear strain differently
}

void GramSchmidtOrthonormalization(inout float3 u, inout float3 v, inout float3 w) {
    float3 originalU = u;
    float3 originalV = v;
    float3 originalW = w;

    u = normalize(u);
    v = v - dot(v, u) * u;

    if (length(v) < 1e-6) {
        v = cross(u, originalW);
        if (length(v) < 1e-6) {
            v = cross(u, float3(0, 1, 0));
        }
    }

    v = normalize(v);
    w = cross(u, v); 
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
    float3 centroidVelocity = float3(0, 0, 0);
    for (int i = 0; i < 8; ++i) {
        centroidPosition += particles[voxel.particleIndices[i]].position;
        centroidVelocity += particles[voxel.particleIndices[i]].velocity;
    }
    centroidPosition /= 8.0;
    centroidVelocity /= 8.0;

    voxel.centroidVelocity = centroidVelocity;

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

    const uint faceIndices[6][4] = {
        {1, 2, 6, 5}, // +X face
        {0, 3, 7, 4}, // -X face
        {2, 3, 7, 6}, // +Y face
        {0, 1, 5, 4}, // -Y face
        {4, 5, 6, 7}, // +Z face
        {0, 1, 2, 3}  // -Z face
    };

    float totalStrain = 0.0;

    for (int face = 0; face < 6; face++) {

        float3 faceCenter;
        float3 expectedNormal;
        float3 faceVelocity;
        float3 shearDir;

        if (voxel.faceConnections[face]) {
            
            faceCenter = float3(0, 0, 0);
            faceVelocity = float3(0, 0, 0);

            for (int fi = 0; fi < 4; fi++) {
                uint pIdx = voxel.particleIndices[faceIndices[face][fi]];
                faceCenter += particles[pIdx].position;
                faceVelocity += particles[pIdx].velocity;
            }
            faceCenter /= 4.0;
            faceVelocity /= 4.0;

            
            
            if (face < 2) {
                expectedNormal = (face == 0) ? u : -u;
                shearDir = v; // Check shear along Y axis
            }
            else if (face < 4) {
                expectedNormal = (face == 2) ? v : -v;
                shearDir = w; // Check shear along Z axis
            }
            else {
                expectedNormal = (face == 4) ? w : -w;
                shearDir = u; // Check shear along X axis
            }
        }

        // Calculate actual face normal and relative velocity
        float3 actualNormal = normalize(faceCenter - centroidPosition);
        float3 relativeVelocity = faceVelocity - centroidVelocity;

        // Calculate enhanced strain including shear and velocity effects
        float strain = CalculateFaceStrain(actualNormal, expectedNormal, shearDir);
        strain += length(relativeVelocity) * 0.1;

        voxel.faceStrains[face] = strain;
        totalStrain += strain;

        // Check for breaking with random variation
        float randomVariation = Random(float2(voxelIndex + face, randomSeed));
        float adjustedThreshold = breakingThreshold * (0.8 + 0.4 * randomVariation);

        if (strain > adjustedThreshold ||
            voxel.accumulatedStrain > adjustedThreshold * 2.0) {
            voxel.faceConnections[face] = false;
        }
    
    }

    voxel.accumulatedStrain = lerp(voxel.accumulatedStrain + totalStrain / 6.0,
        0.0,
        deltaTime / strainMemory);

    // Constraint parameters
    const float constraintStiffness = 1.0f;
    const float dt = 0.033f; // Timestep

    for (int j = 0; j < 8; ++j) {
        uint pIndex = voxel.particleIndices[j];
        Particle p = particles[pIndex];

        // Calculate world position with rotation
        float3 localPos = localPositions[j];
        float3 worldPos = centroidPosition +
            u * localPos.x +
            v * localPos.y +
            w * localPos.z;

        p.prevPosition = p.position;

        // Apply position correction 
        if (p.invMass > 0.0f) {
            float3 correction = worldPos - p.position;

            // Enhanced anisotropic breaking
            float3 correctionScale = float3(1, 1, 1);
            int brokenFaces = 0;

            for (int face = 0; face < 6; face++) {
                if (!voxel.faceConnections[face]) {
                    brokenFaces++;
                    if (face < 2) correctionScale.x *= 0.1;
                    else if (face < 4) correctionScale.y *= 0.1;
                    else correctionScale.z *= 0.1;
                }
            }

            if (brokenFaces > 0 && brokenFaces < 6) {
                float3 rotationalForce = cross(centroidVelocity, worldPos - centroidPosition);
                correction += rotationalForce * rotationalInertia * deltaTime;
            }

            correction *= correctionScale;
            p.position += correction * p.invMass * constraintStiffness;
            p.velocity = (p.position - p.prevPosition) / dt;

        }

        particles[pIndex] = p;
    }

    // Write back updated voxel
    voxels[voxelIndex] = voxel;
}