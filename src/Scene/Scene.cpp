#include "Scene.h"

Scene::Scene(DXContext* p_context, RenderPipeline* p_pipeline) : context(p_context), pipeline(p_pipeline), cmdList(p_pipeline->getCommandList()) {
	inputStrings.push_back("objs\\wolf.obj");
    //inputStrings.push_back("objs\\triangle.obj");
    constructScene();
}

void Scene::constructScene() {
	for (auto string : inputStrings) {
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, cmdList, pipeline);
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

void Scene::draw(ComPointer<ID3D12PipelineState>& pso, ComPointer<ID3D12RootSignature>& rootSignature, Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        cmdList->IASetVertexBuffers(0, 1, m.getVBV());
        cmdList->IASetIndexBuffer(m.getIBV());
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        //NO NEED TO RESET VIEWPORT??
        // == PSO ==
        cmdList->SetPipelineState(pso);
        cmdList->SetGraphicsRootSignature(rootSignature);
        // == ROOT ==

        ID3D12DescriptorHeap* descriptorHeaps[] = { pipeline->getDescriptorHeap().Get() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
        cmdList->SetGraphicsRootDescriptorTable(1, pipeline->getDescriptorHeap()->GetGPUDescriptorHandleForHeapStart()); // Descriptor table slot 1 for SRV

        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &viewMat, 0);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &projMat, 16);

        cmdList->DrawIndexedInstanced(m.getNumTriangles() * 3, 1, 0, 0, 0);
    }
}

size_t Scene::getSceneSize() {
	return sceneSize;
}

void Scene::releaseResources() {
    for (Mesh m : meshes) {
        m.releaseResources();
    }
}
