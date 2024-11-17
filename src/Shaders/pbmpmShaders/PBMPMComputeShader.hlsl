#include "PBMPMComputeRootSignature.hlsl"  // Includes the ROOTSIG definition

// Root constants bound to b0
struct PBMPMConstants {
	float2 gridSize;
	float deltaTime;
	float gravityStrength;
	float liquidRelaxation;
    float liquidViscosity;
	uint fixedPointMultiplier;
	uint particlesPerCellAxis;
	float frictionAngle;
    uint simFrame;
};

ConstantBuffer<PBMPMConstants> cb : register(b0);

// UAV for positions buffer (output buffer)
RWStructuredBuffer<float3> positionsBuffer : register(u0);

// UAV for velocities buffer (output buffer)
RWStructuredBuffer<float3> velocitiesBuffer : register(u1);

float Random(float2 seed) {
    // Apply some mathematical transformations for randomness
    float dotProduct = dot(seed, float2(12.9898, 78.233));
    float sinValue = sin(dotProduct) * 43758.5453;

    // Fract() returns the fractional part to get a pseudo-random number between 0 and 1
    return frac(sinValue);
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
	float deltaTime = cb.deltaTime;
	float2 position = float2(positionsBuffer[DTid.x].x, positionsBuffer[DTid.x].y);
	float2 velocity = float2(velocitiesBuffer[DTid.x].x, velocitiesBuffer[DTid.x].y);
    velocity += float3(0, -1, 0) * cb.gravityStrength * deltaTime;
	position += velocity * deltaTime;

	// Bounce Velocity if the particle hits the ground
	if (abs(position.y) > 0.405f) {
		position.y = sign(position.y) * 0.405f;
		velocity.y = -velocitiesBuffer[DTid.x].y;
		// Randomize the x-velocity a bit
		velocity.x += Random(position) - 0.5;
	}
	// Bounce Velocity if the particle hits the walls
	if (abs(position.x) > 0.76f) {
		position.x = sign(position.x) * 0.76f;
		// Dampen the x-velocity
		velocity.x = -velocitiesBuffer[DTid.x].x * 0.9;
		// Randomize the y-velocity a bit
		velocity.y += Random(position) - 0.5;
	}

	// Update the position and velocity
	positionsBuffer[DTid.x] = float3(position.x, position.y, 0);
	velocitiesBuffer[DTid.x] = float3(velocity.x, velocity.y, 0);
}