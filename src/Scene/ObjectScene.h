#pragma once
#include <filesystem>
#include "Drawable.h"
#include "Mesh.h"

class ObjectScene : public Drawable {
public:
	ObjectScene(DXContext* context, RenderPipeline* pipeline);

	void constructScene();

	void draw(Camera* camera);

	size_t getSceneSize();

	void releaseResources();

private:
	std::vector<std::string> inputStrings;
	std::vector<Mesh> meshes;
	std::vector<XMFLOAT4X4> modelMatrices;

	size_t sceneSize{ 0 };
};