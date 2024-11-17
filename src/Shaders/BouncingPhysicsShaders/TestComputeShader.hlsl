#include "TestComputeRootSignature.hlsl"  // Includes the ROOTSIG definition


struct Particle {
	float3 position;
	float3 prevPosition;
	float3 velocity;
	float invMass;
};

struct DistanceConstraint {
	uint particleA;
	uint particleB;
	float restLength;
};

cbuffer ConstraintParams : register(b0) {
    uint constraintCount;
};

// RWStructuredBuffer for particle data
RWStructuredBuffer<Particle> particles : register(u0);

// StructuredBuffer for constraints
StructuredBuffer<DistanceConstraint> constraints : register(t0);


[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint constraintIndex = DTid.x;
    if (constraintIndex >= constraintCount)
        return;

    // Process the single constraint
    DistanceConstraint dc = constraints[constraintIndex];
    uint idxA = dc.particleA;
    uint idxB = dc.particleB;

    Particle pa = particles[idxA];
    Particle pb = particles[idxB];

    // Calculate correction
    float2 delta = pb.position.xy - pa.position.xy;
    float currentLength = length(delta);
    float2 correction = normalize(delta) * (currentLength - dc.restLength) * 0.5f;

    // Apply correction
    pa.position += float3(correction.xy, 0);
    pb.position -= float3(correction.xy, 0);

    // Write back updated particles
    particles[idxA] = pa;
    particles[idxB] = pb;

}