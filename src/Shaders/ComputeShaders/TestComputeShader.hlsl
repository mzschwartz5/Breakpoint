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

    if (constraintIndex >= 10)
        return;

    DistanceConstraint dc = constraints[constraintIndex];
    Particle pa = particles[dc.particleA];
    Particle pb = particles[dc.particleB];

    float3 delta = pb.position - pa.position;
    float currentLength = length(delta);
    float3 correctionVector = normalize(delta) * ((currentLength - dc.restLength) * 0.1);

    // Apply corrections inversely proportional to mass (invMass)
    if (pa.invMass > 0.0f) {
        pa.position -= correctionVector * pa.invMass;
    }
    if (pb.invMass > 0.0f) {
        pb.position += correctionVector * pb.invMass;
    }

    // Write back corrected positions
    particles[dc.particleA] = pa;
    particles[dc.particleB] = pb;

}