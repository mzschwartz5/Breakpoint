#pragma once

#include "ObjectScene.h"
#include "FluidScene.h"
#include "../D3D/Pipeline/RenderPipeline.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"

enum RenderScene {
	Object,
	Fluid
};

class Scene {
public:
	Scene() = delete;
	Scene(Camera* camera, DXContext* context);

	RenderPipeline* getObjectPipeline();
	MeshPipeline* getMeshPipeline();

	void compute();
	void drawObjects();
	void drawFluid(
		D3D12_GPU_DESCRIPTOR_HANDLE objectColorTextureHandle,
		D3D12_GPU_DESCRIPTOR_HANDLE objectPositionTextureHandle
	);

	DescriptorHeap* getSRVHeap() {
		return fluidScene.getSRVHeapForRenderTextures();
	}

	void releaseResources();

private:
	Camera* camera;

	RenderPipeline objectRP;
	ObjectScene objectScene;
	
	RenderPipeline fluidRP;
	ComputePipeline bilevelUniformGridCP;
	ComputePipeline surfaceBlockDetectionCP;
	ComputePipeline surfaceCellDetectionCP;
	ComputePipeline surfaceVertexCompactionCP;
	ComputePipeline surfaceVertexDensityCP;
	ComputePipeline surfaceVertexNormalCP;
	ComputePipeline bufferClearCP;
	ComputePipeline dispatchArgDivideCP;
	MeshPipeline fluidMeshPipeline;
	FluidScene fluidScene;
};
