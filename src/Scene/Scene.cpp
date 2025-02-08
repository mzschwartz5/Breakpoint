#include "Scene.h"

Scene::Scene(Camera* p_camera, DXContext* context)
	: camera(p_camera),
	objectRP("VertexShader.cso", "PixelShader.cso", "RootSignature.cso", *context, CommandListID::OBJECT_RENDER_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	objectScene(context, &objectRP),
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
	bufferClearCP("bufferClearRootSignature.cso", "bufferClearComputeShader.cso", *context, CommandListID::FLUID_BUFFER_CLEAR_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	dispatchArgDivideCP("DispatchArgDivideRootSig.cso", "DispatchArgDivide.cso", *context, CommandListID::FLUID_DISPATCH_ARG_DIVIDE_COMPUTE_ID,
		D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
	fluidScene(context, &fluidRP, &bilevelUniformGridCP, &surfaceBlockDetectionCP, &surfaceCellDetectionCP, &surfaceVertexCompactionCP, &surfaceVertexDensityCP, &surfaceVertexNormalCP, &bufferClearCP, &dispatchArgDivideCP, &fluidMeshPipeline)
{
}

RenderPipeline* Scene::getObjectPipeline() {
	return &objectRP;
}

MeshPipeline* Scene::getMeshPipeline() {
	return &fluidMeshPipeline;
}

void Scene::compute() {
	fluidScene.compute();
}

void Scene::drawObjects() {
	objectScene.draw(camera);
}

void Scene::drawFluid(
	D3D12_GPU_DESCRIPTOR_HANDLE objectColorTextureHandle,
	D3D12_GPU_DESCRIPTOR_HANDLE objectPositionTextureHandle,
	int screenWidth,
    int screenHeight
) {
	fluidScene.draw(camera, objectColorTextureHandle, objectPositionTextureHandle, screenWidth, screenHeight);
}

void Scene::releaseResources() {
	objectScene.releaseResources();
	fluidScene.releaseResources();
}