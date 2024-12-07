#include "Scene.h"

Scene::Scene(RenderScene p_scene, Camera* p_camera, DXContext* context)
	: scene(p_scene), camera(p_camera),
	objectRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::OBJECT_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	objectScene(context, &objectRP),
	pbmpmRP("PBMPMVertexShader.cso", "PixelShader.cso", "PBMPMVertexRootSignature.cso", *context, CommandListID::PBMPM_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	pbmpmIC(100),
	pbmpmScene(context, &pbmpmRP, pbmpmIC),
	fluidRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::FLUID_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	bilevelUniformGridCP("BilevelUniformGridRootSig.cso", "BilevelUniformGrid.cso", *context, CommandListID::BILEVEL_UNIFORM_GRID_COMPUTE_ID, 
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 45, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceBlockDetectionCP("SurfaceBlockDetectionRootSig.cso", "SurfaceBlockDetection.cso", *context, CommandListID::SURFACE_BLOCK_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceCellDetectionCP("SurfaceCellDetectionRootSig.cso", "SurfaceCellDetection.cso", *context, CommandListID::SURFACE_CELL_DETECTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexCompactionCP("SurfaceVertexCompactionRootSig.cso", "SurfaceVertexCompaction.cso", *context, CommandListID::SURFACE_VERTEX_COMPACTION_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexDensityCP("SurfaceVertexDensityRootSig.cso", "SurfaceVertexDensity.cso", *context, CommandListID::SURFACE_VERTEX_DENSITY_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	surfaceVertexNormalCP("SurfaceVertexNormalsRootSig.cso", "SurfaceVertexNormals.cso", *context, CommandListID::SURFACE_VERTEX_NORMAL_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidMeshPipeline("FluidMeshShader.cso", "FluidSurfaceShader.cso", "FluidMeshRootSig.cso", *context, CommandListID::FLUID_MESH_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidScene(context, &fluidRP, &bilevelUniformGridCP, &surfaceBlockDetectionCP, &surfaceCellDetectionCP, &surfaceVertexCompactionCP, &surfaceVertexDensityCP, &surfaceVertexNormalCP, &fluidMeshPipeline),
	currentRP(),
	currentCP()
{
	setRenderScene(p_scene);
}


RenderPipeline* Scene::getRenderPipeline() {
	return currentRP;
}

MeshPipeline* Scene::getMeshPipeline() {
	return &fluidMeshPipeline;
}

void Scene::setRenderScene(RenderScene renderScene) {
	scene = renderScene;

	switch (scene) {
	case PBMPM:
		currentRP = &pbmpmRP;
		//currentCP = &pbmpmCP;
		break;
	case Fluid:
		currentRP = &fluidRP;
		currentCP = &bilevelUniformGridCP;
		break;
	case Object:
		currentRP = &objectRP;
		currentCP = nullptr;
		break;
	}
}

void Scene::compute() {
	pbmpmScene.compute();
	fluidScene.compute(
		pbmpmScene.getPositionBuffer(),
		pbmpmScene.getParticleCount()
	);
}

void Scene::draw() {
	switch (scene) {
	case PBMPM:
		pbmpmScene.draw(camera);
		break;
	case Object:
		objectScene.draw(camera);
		break;
	}
}

void Scene::drawFluid() {
	fluidScene.draw(camera);
}

void Scene::releaseResources() {
	objectScene.releaseResources();
	pbmpmScene.releaseResources();
	fluidScene.releaseResources();
}

void Scene::updatePBMPMConstants(PBMPMConstants& newConstants) {
	pbmpmScene.updateConstants(newConstants);
}
