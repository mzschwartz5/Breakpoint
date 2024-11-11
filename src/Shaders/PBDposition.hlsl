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
    float stiffness;
};

RWStructuredBuffer<Particle> particles : register(u0);
//StructuredBuffer<DistanceConstraint> constraints : register(t0);


[numthreads(256, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    //uint constraintIndex = id.x;
    //if (constraintIndex >= constraints.Length)
    //    return;

    //DistanceConstraint dc = constraints[constraintIndex];

    //Particle pa = particles[dc.particleA];
    //Particle pb = particles[dc.particleB];

    //float3 delta = pb.position - pa.position;
    //float currentLength = length(delta);
    //float3 n = delta / currentLength;

    //float correction = (currentLength - dc.restLength) / (pa.mass + pb.mass);
    //float3 correctionA = n * correction * pb.mass * stiffness;
    //float3 correctionB = -n * correction * pa.mass * stiffness;

    //// Apply corrections
    //pa.position += correctionA;
    //pb.position += correctionB;

    //// Update particles
    //particles[dc.particleA] = pa;
    //particles[dc.particleB] = pb;
}