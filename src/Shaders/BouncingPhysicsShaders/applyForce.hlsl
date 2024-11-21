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
float Random(float2 seed) {
    // Apply some mathematical transformations for randomness
    float dotProduct = dot(seed, float2(12.9898, 78.233));
    float sinValue = sin(dotProduct) * 43758.5453;

    // Fract() returns the fractional part to get a pseudo-random number between 0 and 1
    return frac(sinValue);
}
[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint particleIndex = DTid.x;
    if (particleIndex >= 16)
        return;

    Particle p = particles[particleIndex];

    // Save current position as previous position
    p.prevPosition = p.position;

    // Update velocity with gravity
    p.velocity += float3(0.0, -9.81f * 0.01f * 0.033, 0.0) ;

    // Predict new position
    p.position += float3(p.velocity.xy * 0.033, 0.0);

    float boundaryX = 0.72f;
    float boundaryY = 0.365f;
    float restitution = 0.9f;

    if (p.position.y < -boundaryY) {
        p.position.y = -boundaryY;
        p.velocity.y = -p.velocity.y * restitution;
    }
    else if (p.position.y > boundaryY) {
        p.position.y = boundaryY;
        p.velocity.y = -p.velocity.y * restitution;
    }

    // Collision with X boundaries
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
