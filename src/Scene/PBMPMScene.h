#pragma once

#include "../D3D/DXContext.h"
#include "Drawable.h"
#include "../D3D/StructuredBuffer.h"
#include "../D3D/VertexBuffer.h"
#include "../D3D/IndexBuffer.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "Geometry.h"
#include <math.h>

// Keep consistent with PBMPMCommon.hlsl

const unsigned int ParticleDispatchSize = 64;
const unsigned int GridDispatchSize = 8;
const unsigned int BukkitSize = 6;
const unsigned int BukkitHaloSize = 1;
const unsigned int GuardianSize = 3;

const unsigned int maxParticles = 50000;
const unsigned int maxTimestampCount = 2048;

struct PBMPMConstants {
	XMUINT2 gridSize;
	float deltaTime;
	float gravityStrength;

	float liquidRelaxation;
	float liquidViscosity;
	unsigned int fixedPointMultiplier;

	unsigned int useGridVolumeForLiquid;
	unsigned int particlesPerCellAxis;

	float frictionAngle;
	unsigned int shapeCount;
	unsigned int simFrame;

	unsigned int bukkitCount;
	unsigned int bukkitCountX;
	unsigned int bukkitCountY;
	unsigned int iteration;
	unsigned int iterationCount;
	float borderFriction;
};

struct ShapeFactory {
	XMFLOAT2 position;
	XMFLOAT2 halfSize;

	float radius;
	float rotation;
	float functionality;
	float shapeType;

	float emitMaterial;
	float emissionRate;
	float emissionSpeed;
	float padding;
};

struct PBMPMParticle {
	XMFLOAT2 position;
	XMFLOAT2 displacement;
	XMFLOAT4 deformationGradient;
	XMFLOAT4 deformationDisplacement;

	float liquidDensity;
	float mass;
	float material;
	float volume;
	
	float lambda;
	float logJp;
	float enabled;
};

struct BukkitSystem {
	unsigned int countX;
	unsigned int countY;
	unsigned int count;
	StructuredBuffer countBuffer;
	StructuredBuffer countBuffer2;
	StructuredBuffer particleData;
	StructuredBuffer threadData;
	StructuredBuffer dispatch;
	StructuredBuffer blankDispatch;
	StructuredBuffer particleAllocator;
	StructuredBuffer indexStart;
};

struct BukkitThreadData {
	unsigned int rangeStart;
	unsigned int rangeCount;
	unsigned int bukkitX;
	unsigned int bukkitY;
};

class PBMPMScene : public Drawable {
public:
	PBMPMScene(DXContext* context, RenderPipeline* renderPipeline, unsigned int instanceCount);

	void constructScene();

	void compute();

	void draw(Camera* camera);

	void releaseResources();

	void updateConstants(PBMPMConstants& newConstants);

private:
	DXContext* context;
	RenderPipeline* renderPipeline;

	ComputePipeline g2p2gPipeline;
	ComputePipeline bukkitCountPipeline;
	ComputePipeline bukkitAllocatePipeline;
	ComputePipeline bukkitInsertPipeline;
	ComputePipeline bufferClearPipeline;

	PBMPMConstants constants;
	BukkitSystem bukkitSystem;

	XMMATRIX modelMat;
	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	unsigned int instanceCount;
	ID3D12CommandSignature* commandSignature = nullptr;
	UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;
	unsigned int indexCount = 0;

	unsigned int substepIndex = 0;

	// Scene Buffers
	StructuredBuffer particleBuffer;
	StructuredBuffer particleFreeIndicesBuffer;
	StructuredBuffer particleCount;
	StructuredBuffer particleSimDispatch;

	std::array<StructuredBuffer, 3> gridBuffers;

	void createBukkitSystem();

	void updateSimUniforms(unsigned int iteration);

	void resetBuffers(bool resetGrids = false);

	void bukkitizeParticles();
};