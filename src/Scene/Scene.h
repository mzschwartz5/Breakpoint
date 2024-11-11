#pragma once

#include <filesystem>

#include "Scene/Mesh.h"
#include "Scene/Camera.h"

#include "D3D/Pipeline/RenderPipeline.h"

class Scene {
public:
	Scene(DXContext* context, RenderPipeline* pipeline, ID3D12GraphicsCommandList5* cmdList);

	void constructScene();

	void draw(Camera* camera);

	size_t getSceneSize();

	void releaseResources();

private:
	std::vector<std::string> inputStrings;
	std::vector<Mesh> meshes;

	size_t sceneSize{ 0 };

	DXContext* context;
	RenderPipeline* pipeline;
	ID3D12GraphicsCommandList5* cmdList;
};