#include "VelocitySignature.hlsl"

struct Particle {
    float3 position;
    float3 prevPosition;
    float3 velocity;
    float invMass;
};

cbuffer SimulationParams : register(b0) {
    float deltaTime;
    float count;
    float3 gravity;
};

RWStructuredBuffer<Particle> particles : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIndex = DTid.x;
    if (particleIndex >= count)
        return;

    Particle p = particles[particleIndex];

    // Save current position as previous position
    p.prevPosition = p.position;

    // Update velocity with gravity
    p.velocity += gravity * deltaTime;

    // Predict new position
    p.position += p.velocity * deltaTime;

    if (abs(p.position.y) > 0.365f) {
        p.position.y = sign(p.position.y) * 0.365f;

    }

    if (abs(p.position.x) > 0.72f) {
        p.position.x = sign(p.position.x) * 0.72f;

    }



    particles[particleIndex] = p;
}
