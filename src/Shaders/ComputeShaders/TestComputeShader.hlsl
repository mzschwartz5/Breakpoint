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

//float Random(float2 seed) {
//    // Apply some mathematical transformations for randomness
//    float dotProduct = dot(seed, float2(12.9898, 78.233));
//    float sinValue = sin(dotProduct) * 43758.5453;
//
//    // Fract() returns the fractional part to get a pseudo-random number between 0 and 1
//    return frac(sinValue);
//}





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
    float3 delta = pb.position - pa.position;
    float currentLength = length(delta);
    float3 correction = normalize(delta) * (currentLength - dc.restLength) * 0.5f;

    // Apply correction
    pa.position += correction;
    pb.position -= correction;

    // Write back updated particles
    particles[idxA] = pa;
    particles[idxB] = pb;

}