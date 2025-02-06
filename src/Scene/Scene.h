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

	RenderPipeline* getRenderPipeline();
	MeshPipeline* getMeshPipeline();

	void compute();
	void draw();
	void drawFluid();

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
