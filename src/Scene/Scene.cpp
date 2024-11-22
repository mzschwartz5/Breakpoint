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
	fluidRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::BILEVEL_UNIFORM_GRID_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bilevelUniformGridCP("BilevelUniformGridRootSig.cso", "BilevelUniformGrid.cso", *context, CommandListID::BILEVEL_UNIFORM_GRID_COMPUTE_ID, 
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceBlockDetectionCP("SurfaceBlockDetectionRootSig.cso", "SurfaceBlockDetection.cso", *context, CommandListID::SURFACE_BLOCK_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceCellDetectionCP("SurfaceCellDetectionRootSig.cso", "SurfaceCellDetection.cso", *context, CommandListID::SURFACE_CELL_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexCompactionCP("SurfaceVertexCompactionRootSig.cso", "SurfaceVertexCompaction.cso", *context, CommandListID::SURFACE_VERTEX_COMPACTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexDensityCP("SurfaceVertexDensityRootSig.cso", "SurfaceVertexDensity.cso", *context, CommandListID::SURFACE_VERTEX_DENSITY_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidScene(context, &fluidRP, &bilevelUniformGridCP, &surfaceBlockDetectionCP, &surfaceCellDetectionCP, &surfaceVertexCompactionCP, &surfaceVertexDensityCP),
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
	case Fluid:
		currentRP = &fluidRP;
		currentCP = &bilevelUniformGridCP;
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
	case Fluid:
		fluidScene.compute();
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
	case Fluid:
		fluidScene.draw(camera);
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
	fluidScene.releaseResources();
}
