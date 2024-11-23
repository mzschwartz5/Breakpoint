#include "VelocitySignature.hlsl"

struct Particle {
    float3 position;
    float3 previousPosition;
    float3 velocity;
    float inverseMass;
};

cbuffer SimulationParams : register(b0) {
    uint constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float3 gravity;
};

RWStructuredBuffer<Particle> particles : register(u0);


[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIndex = DTid.x;
    if (particleIndex >= count)
        return;

    Particle p = particles[particleIndex];

    // Update velocity based on position change
    p.velocity = (p.position - p.previousPosition) / 0.33f;

    particles[particleIndex] = p;
}
