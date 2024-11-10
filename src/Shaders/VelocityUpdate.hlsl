#include "RootSignature.hlsl"

struct Particle {
    float3 position;
    float3 previousPosition;
    float3 velocity;
    float inverseMass;
};

cbuffer SimulationParams : register(b0) {
    float deltaTime;
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(ROOTSIG)]
[numthreads(256, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    if (id >= particles.Length) return;

    Particle p = particles[id];

    if (p.inverseMass > 0.0f) {
       
        p.velocity = (p.position - p.previousPosition) / deltaTime;

        float damping = 0.98f; // Adjust damping factor as needed
        p.velocity *= damping;
    }
    else {
        p.velocity = float3(0.0f, 0.0f, 0.0f);
    }

    particles[id] = p;
}
