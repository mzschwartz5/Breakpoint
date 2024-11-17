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
    if (particleIndex >= 2)
        return;

    Particle p = particles[particleIndex];

    // Save current position as previous position
    p.prevPosition = p.position;

    // Update velocity with gravity
    p.velocity += float3(0, - 9.81f * 0.1f * 0.33, 0);

    // Predict new position
    p.position += float3(p.velocity.xy * 0.33, 0);

    if (abs(p.position.y) > 0.365f) {
        p.position.y = sign(p.position.y) * 0.365f;
        p.velocity.y = -p.velocity.y;
       
        p.velocity.x += Random(p.position) - 0.5;
    }

    if (abs(p.position.x) > 0.72f) {
        p.position.x = sign(p.position.x) * 0.72f;
        p.velocity.x = -p.velocity.x * 0.9;
        // Randomize the y-velocity a bit
        p.velocity.y += Random(p.position) - 0.5;
    }

    particles[particleIndex] = p;
}
