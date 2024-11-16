#include "Scene.h"

Scene::Scene(RenderScene p_scene, Camera* p_camera, DXContext* context) 
	: scene(p_scene), camera(p_camera),
	objectRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::OBJECT_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	objectScene(context, &objectRP),
	pbmpmRP("PBMPMVertexShader.cso", "PixelShader.cso", "PBMPMVertexRootSignature.cso", *context, CommandListID::PBMPM_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	pbmpmCP("PBMPMComputeRootSignature.cso", "PBMPMComputeShader.cso", *context, CommandListID::PBMPM_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	pbmpmIC(50),
	pbmpmScene(context, &pbmpmRP, &pbmpmCP, pbmpmIC),
	physicsRP("PhysicsVertexShader.cso", "PixelShader.cso", "PhysicsRootSignature.cso", *context, CommandListID::PHYSICS_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	physicsCP("TestComputeRootSignature.cso", "TestComputeShader.cso", *context, CommandListID::PHYSICS_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	physicsIC(10),
	physicsScene(context, &physicsRP, &physicsCP, physicsIC),
	currentRP(&objectRP),
	currentCP(nullptr)
{}

RenderPipeline* Scene::getRenderPipeline() {
	return currentRP;
}

void Scene::setRenderScene(RenderScene renderScene) {
	scene = renderScene;

	switch (scene) {
	case PBMPM:
		currentRP = &pbmpmRP;
		currentCP = &pbmpmCP;
		break;
	case Physics:
		currentRP = &physicsRP;
		currentCP = &physicsCP;
		break;
	case Object:
	default:
		currentRP = &objectRP;
		currentCP = nullptr;
		break;
	}
}

void Scene::compute() {
	switch (scene) {
	case PBMPM:
		pbmpmScene.compute();
		break;
	case Physics:
		physicsScene.compute();
		break;
	default:
		break;
	}
}

void Scene::draw() {
	switch (scene) {
	case Physics:
		physicsScene.draw(camera);
		break;
	case PBMPM:
		pbmpmScene.draw(camera);
		break;
	default:
	case Object:
		objectScene.draw(camera);
		break;
	}
}

void Scene::releaseResources() {
	objectScene.releaseResources();
	pbmpmScene.releaseResources();
	physicsScene.releaseResources();
}
