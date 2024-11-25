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

[numthreads(256, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIndex = DTid.x;
    if (particleIndex >= count) return;

    Particle p = particles[particleIndex];
    float h = deltaTime / numSubsteps;

    // Save current position as previous position
    p.prevPosition = p.position;

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