#include "Scene.h"

Scene::Scene(DXContext* p_context, ID3D12GraphicsCommandList5* p_cmdList) : context(p_context), cmdList(p_cmdList) {
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

void Scene::draw(RenderPipeline& pipeline, ComPointer<ID3D12PipelineState>& pso, ComPointer<ID3D12RootSignature>& rootSignature, Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        cmdList->IASetVertexBuffers(0, 1, &m.getVBV());
        cmdList->IASetIndexBuffer(&m.getIBV());
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        //NO NEED TO RESET VIEWPORT??
        // == PSO ==
        cmdList->SetPipelineState(pso);
        cmdList->SetGraphicsRootSignature(rootSignature);
        // == ROOT ==

        ID3D12DescriptorHeap* descriptorHeaps[] = { pipeline.getSrvHeap().Get() };
        cmdList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);
        cmdList->SetGraphicsRootDescriptorTable(1, pipeline.getSrvHeap()->GetGPUDescriptorHandleForHeapStart()); // Descriptor table slot 1 for SRV

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
