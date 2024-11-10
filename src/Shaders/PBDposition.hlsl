//#define ROOTSIG \
//"DescriptorTable(UAV(u0)), "                    /* UAV for particles */ \
//"RootConstants(num32BitConstants=4, b0)"        /* deltaTime and gravity */

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

[RootSignature(ROOTSIG)]
[numthreads(256, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    if (id >= particles.Length) return;

    Particle p = particles[id];

    if (p.inverseMass > 0.0f) {
        // Apply external forces: should be changed later
        float3 acceleration = gravity;

        
        p.velocity += acceleration * deltaTime;

        float3 tempPosition = p.position;

        p.position += p.velocity * deltaTime;

        p.previousPosition = tempPosition;
    }

    particles[id] = p;
}