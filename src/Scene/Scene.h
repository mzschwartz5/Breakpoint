#pragma once

#include <filesystem>

#include "Scene/Mesh.h"

class Scene {
public:
	Scene(DXContext* context, ID3D12GraphicsCommandList5* cmdList);
	void constructScene(DXContext* context, ID3D12GraphicsCommandList5* cmdList);

	size_t getSceneSize();

private:
	std::vector<std::string> inputStrings;
	std::vector<Mesh> meshes;

	size_t sceneSize{ 0 };
};