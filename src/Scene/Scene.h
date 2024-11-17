#pragma once

#include "Scene/Camera.h"

#include "D3D/Pipeline/RenderPipeline.h"

class Scene {
public:
	Scene(DXContext* context, RenderPipeline* pipeline);

	void constructScene();

	void draw(Camera* camera);

	void releaseResources();

protected:
	DXContext* context;
	RenderPipeline* pipeline;
};