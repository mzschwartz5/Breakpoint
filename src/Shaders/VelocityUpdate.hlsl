#include "RootSignature.hlsl"

struct Particle {
    float3 position;
    float3 previousPosition;
    float3 velocity;
    float inverseMass;
};

cbuffer SimulationParams : register(b0) {
    float deltaTime;
    float3 gravity;
};

RWStructuredBuffer<Particle> particles : register(u0);


[numthreads(256, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    //uint particleIndex = DTid.x;
    //if (particleIndex >= particles.Length)
    //    return;

    //Particle p = particles[particleIndex];

    //// Update velocity with gravity
    //p.velocity += gravity * deltaTime;

    //// Update position with velocity
    //p.position += p.velocity * deltaTime;

    //particles[particleIndex] = p;
}
