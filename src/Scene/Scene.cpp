#include "Scene.h"

Scene::Scene(DXContext* context, ID3D12GraphicsCommandList5* cmdList) {
	inputStrings.push_back("objs\\wolf.obj");
	constructScene(context, cmdList);
}

void Scene::constructScene(DXContext* context, ID3D12GraphicsCommandList5* cmdList) {
	for (auto string : inputStrings) {
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, cmdList);
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

size_t Scene::getSceneSize() {
	return sceneSize;
}
