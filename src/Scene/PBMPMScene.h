#pragma once

#include "Scene.h"
#include "../D3D/StructuredBuffer.h"
#include "../D3D/VertexBuffer.h"
#include "../D3D/IndexBuffer.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "Geometry.h"

struct PBMPMConstants {
	float gravityStrength;
	float inputX;
	float inputY;
	float deltaTime;
};

class PBMPMScene : public Scene {
public:
	PBMPMScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* compPipeline, unsigned int instanceCount);

	void constructScene();

	void compute();

	void draw(Camera* camera);

	void releaseResources();

private:
	DXContext* context;
	RenderPipeline* pipeline;
	ComputePipeline* computePipeline;
	XMMATRIX modelMat;
	std::vector<XMFLOAT3> positions;
	std::vector<XMFLOAT3> velocities;
	StructuredBuffer positionBuffer;
	StructuredBuffer velocityBuffer;
	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	unsigned int instanceCount;
	UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;
	unsigned int indexCount = 0;
};