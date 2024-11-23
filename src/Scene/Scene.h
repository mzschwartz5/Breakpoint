#pragma once

#include "ObjectScene.h"
#include "PBMPMScene.h"
#include "PhysicsScene.h"
#include "FluidScene.h"
#include "../D3D/Pipeline/RenderPipeline.h"
#include "../D3D/Pipeline/ComputePipeline.h"
#include "../D3D/Pipeline/MeshPipeline.h"

enum RenderScene {
	Object,
	PBMPM,
	Physics,
	Fluid
};

class Scene {
public:
	Scene() = delete;
	Scene(RenderScene renderScene, Camera* camera, DXContext* context);

	RenderPipeline* getRenderPipeline();

	void setRenderScene(RenderScene renderScene);
	void compute();
	void draw();

	void releaseResources();

private:
	Camera* camera;

	RenderPipeline objectRP;
	ObjectScene objectScene;
	
	RenderPipeline pbmpmRP;
	ComputePipeline pbmpmCP;
	unsigned int pbmpmIC;
	PBMPMScene pbmpmScene;

	RenderPipeline physicsRP; 
	ComputePipeline physicsCP;
	unsigned int physicsIC;
	PhysicsScene physicsScene;

	RenderPipeline fluidRP;
	ComputePipeline bilevelUniformGridCP;
	ComputePipeline surfaceBlockDetectionCP;
	ComputePipeline surfaceCellDetectionCP;
	ComputePipeline surfaceVertexCompactionCP;
	ComputePipeline surfaceVertexDensityCP;
	ComputePipeline surfaceVertexNormalCP;
	FluidScene fluidScene;

	RenderScene scene;

	RenderPipeline* currentRP;
	ComputePipeline* currentCP;
};
