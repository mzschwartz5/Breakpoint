#include "g2p2gRootSignature.hlsl"  // Includes the ROOTSIG definition

// Root constants bound to b0
struct PBMPMConstants {
	float2 gridSize;
	float deltaTime;
	float gravityStrength;

	float liquidRelaxation;
    float liquidViscosity;
	uint fixedPointMultiplier;

	uint useGridVolumeForLiquid
	uint particlesPerCellAxis;

	float frictionAngle;
	unsigned int shapeCount;
    uint simFrame;

	uint bukkitCount;
	uint bukkitCountX;
	uint bukkitCountY;
	uint iteration;
	uint iterationCount;
	uint borderFriction;

	uint TotalBukkitEdgeLength;
	uint TileDataSizePerEdge;
	uint TileDataSize;
};

ConstantBuffer<PBMPMConstants> cb : register(b0);

struct Particle {
	float2 position;
	float2 velocity;
	// Additional fields as needed
};

struct BukkitThreadData {
	int someField;
	// Additional fields as needed
};

// Structured Buffer for particles (read-write UAV)
RWStructuredBuffer<Particle> g_particles : register(u0);

// Structured Buffer for grid source data (read-only SRV)
StructuredBuffer<int> g_gridSrc : register(t0);

// Structured Buffer for grid destination data (read-write UAV with atomic support)
RWStructuredBuffer<int> g_gridDst : register(u1);

// Structured Buffer for grid cells to be cleared (read-write UAV)
RWStructuredBuffer<int> g_gridToBeCleared : register(u2);

// Structured Buffer for bukkit thread data (read-only SRV)
StructuredBuffer<BukkitThreadData> g_bukkitThreadData : register(t1);

// Structured Buffer for bukkit particle indices (read-only SRV)
StructuredBuffer<uint> g_bukkitParticleData : register(t2);

// Structured Buffer for simulation shapes (read-only SRV)
StructuredBuffer<SimShape> g_shapes : register(t3);

// Structured Buffer for free indices with atomic access (read-write UAV)
RWStructuredBuffer<int> g_freeIndices : register(u3);

groupshared int s_tileData[TileDataSize];
groupshared int s_tileDataDst[TileDataSize];

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