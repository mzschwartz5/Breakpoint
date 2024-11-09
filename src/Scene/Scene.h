#pragma once

#include <filesystem>

#include "Scene/Mesh.h"
#include "Scene/Camera.h"

#include "D3D/RenderPipeline.h"

class Scene {
public:
	Scene(DXContext* context, ID3D12GraphicsCommandList5* cmdList);

	void constructScene(DXContext* context, ID3D12GraphicsCommandList5* cmdList);

	void draw(RenderPipeline& pipeline, ComPointer<ID3D12PipelineState>& pso, ComPointer<ID3D12RootSignature>& rootSignature, Camera* camera);

	size_t getSceneSize();

private:
	std::vector<std::string> inputStrings;
	std::vector<Mesh> meshes;

	size_t sceneSize{ 0 };

	DXContext* context;
	ID3D12GraphicsCommandList5* cmdList;


};