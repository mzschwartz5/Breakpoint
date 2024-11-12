#include "ObjectScene.h"

ObjectScene::ObjectScene(DXContext* context, RenderPipeline* pipeline)
	: Scene(context, pipeline)
{
	constructScene();
}

void ObjectScene::constructScene()
{
	pipeline->createPSOD();
	pipeline->createPipelineState(context->getDevice());
	inputStrings.push_back("objs\\wolf.obj");
	for (auto string : inputStrings) {
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, pipeline->getCommandList(), pipeline);
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

void ObjectScene::draw(Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        auto cmdList = pipeline->getCommandList();
        cmdList->IASetVertexBuffers(0, 1, m.getVBV());
        cmdList->IASetIndexBuffer(m.getIBV());
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        //NO NEED TO RESET VIEWPORT??
        // == PSO ==
        cmdList->SetPipelineState(pipeline->getPSO());
        cmdList->SetGraphicsRootSignature(pipeline->getRootSignature());
        // == ROOT ==

        ID3D12DescriptorHeap* descriptorHeaps[] = { pipeline->getDescriptorHeap()->GetAddress() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
        cmdList->SetGraphicsRootDescriptorTable(1, pipeline->getDescriptorHeap()->GetGPUHandleAt(0)); // Descriptor table slot 1 for SRV

        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);

        cmdList->DrawIndexedInstanced(m.getNumTriangles() * 3, 1, 0, 0, 0);
    }
}

size_t ObjectScene::getSceneSize() {
    return sceneSize;
}

void ObjectScene::releaseResources() {
	for (Mesh m : meshes) {
		m.releaseResources();
	}
	pipeline->releaseResources();
}