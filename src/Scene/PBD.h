#pragma once

#include "Scene.h"
#include "../D3D/StructuredBuffer.h"
#include "../D3D/VertexBuffer.h"
#include "../D3D/IndexBuffer.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "Geometry.h"
#include "./PBD/particles.h"
#include <chrono>


class PBDScene : public Scene {
public:
	PBDScene(DXContext* context, RenderPipeline* pipeline, ComputePipeline* computePipeline, 
		ComputePipeline* applyForcesPipeline, ComputePipeline* velocityUpdatePipeline, ComputePipeline* FaceToFacePipeline, unsigned int instances);

	/*void testBreaking(std::vector<Particle> particles);
	void testTwisting(std::vector<Particle> particles);*/
	void testVoxels(std::vector<Particle>* particles,
		std::vector<Voxel>* voxels);

	void createPartitions(
		std::vector<std::vector<uint32_t>>* partitionIndices,
		std::vector<Particle>* particles,
		std::vector<Voxel>* voxels,
		ComputePipeline* computePipeline);

	void constructScene();

	void compute();

	void draw(Camera* camera);

	void releaseResources();

private:
	DXContext* context;
	RenderPipeline* pipeline;
	ComputePipeline* computePipeline;
	ComputePipeline* applyForcesPipeline;
	ComputePipeline* velocityUpdatePipeline;
	ComputePipeline* FaceToFacePipeline;
	XMMATRIX modelMat;


	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	unsigned int instanceCount;
	UINT64 fenceValue = 1;
	ComPointer<ID3D12Fence> fence;
	unsigned int indexCount = 0;

	SimulationParams simParams = {};
	unsigned int constraintCount = 0;

	StructuredBuffer particleBuffer;
	StructuredBuffer voxelBuffer;
	std::vector<Particle> particles;
	std::vector<Voxel> voxels;
	
	StructuredBuffer shapePartitionBuffer;    // Partition 0
	StructuredBuffer xFacePartitionBuffer;    // Partition 1
	StructuredBuffer yFacePartitionBuffer;    // Partition 2
	StructuredBuffer zFacePartitionBuffer;    // Partition 3

	std::vector<std::vector<uint32_t>> partitionIndices;


	

};