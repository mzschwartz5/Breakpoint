#include "VelocitySignature.hlsl"

struct Particle {
    float3 position;
    float3 prevPosition;
    float3 velocity;
    float invMass;
};

cbuffer SimulationParams : register(b0) {
    uint constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float3 gravity;
    float compliance;
    float numSubsteps;
};

RWStructuredBuffer<Particle> particles : register(u0);

float Random(float2 seed) {
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
}

float3 RandomDirection(float2 seed) {
    float theta = Random(seed) * 2.0 * 3.14159;
    float phi = Random(seed + float2(1.0, 1.0)) * 3.14159;

    float3 dir;
    dir.x = sin(phi) * cos(theta);
    dir.y = sin(phi) * sin(theta);
    dir.z = cos(phi);

    return normalize(dir);
}

float3 TestBreaking(uint particleIndex, inout Particle p) {
    // Check if particle belongs to second voxel (indices 8-15)
    if (particleIndex >= 8 && particleIndex < 16) {

        // Apply strong impulse to break the voxel away
        float breakingForce = 10.0f;  // Adjust this value to control breaking force
        float3 breakingDirection = float3(1.0f, 0.5f, 0.0f);  // Diagonal upward-right direction

       
        float2 seed = float2(particleIndex, randomSeed);
        float3 randomDir = RandomDirection(seed) * 0.2f;  // 20% random variation
        breakingDirection = normalize(breakingDirection + randomDir);

        // Apply the breaking impulse
       return breakingDirection * breakingForce;
    }
    return float3(0.0, 0.0, 0.0);
}

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIndex = DTid.x;
    if (particleIndex >= count) return;

    Particle p = particles[particleIndex];
    float h = deltaTime / numSubsteps;

    // Save current position as previous position
    p.prevPosition = p.position;

    // Test breaking - apply impulse to second voxel
    //p.velocity += TestBreaking(particleIndex, p) * h;

    // Apply gravity and update velocity
    p.velocity += gravity * h;

    // Predict new position
    p.position += p.velocity * h;

    // Boundary collision handling
    float boundaryX = 0.72f;
    float boundaryY = 0.365f;
    float restitution = 0.9f;
    // Y boundaries
    if (p.position.y < -boundaryY) {
        p.position.y = -boundaryY;
        p.velocity.y = -p.velocity.y * restitution;
    }
    else if (p.position.y > boundaryY) {
        p.position.y = boundaryY;
        p.velocity.y = -p.velocity.y * restitution;
    }

    // X boundaries
    if (p.position.x < -boundaryX) {
        p.position.x = -boundaryX;
        p.velocity.x = -p.velocity.x * restitution;
    }
    else if (p.position.x > boundaryX) {
        p.position.x = boundaryX;
        p.velocity.x = -p.velocity.x * restitution;
    }

    particles[particleIndex] = p;
}