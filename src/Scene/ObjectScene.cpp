#include "ObjectScene.h"

ObjectScene::ObjectScene(DXContext* context, RenderPipeline* pipeline)
	: Drawable(context, pipeline)
{
	constructScene();
}

void ObjectScene::constructScene()
{
	renderPipeline->createPSOD();
	renderPipeline->createPipelineState(context->getDevice());

	inputStrings.push_back("objs\\Beach.obj");

    XMFLOAT4X4 m1;
    XMStoreFloat4x4(&m1, XMMatrixTranslation(0, 0, 0));
    modelMatrices.push_back(m1);

    XMFLOAT4X4 m2;
    XMStoreFloat4x4(&m2, XMMatrixScaling(0.2f, 0.2f, 0.2f) * XMMatrixTranslation(-10, -8, 0));
    modelMatrices.push_back(m2);

    for (int i = 0; i < inputStrings.size(); i++) {
        auto string = inputStrings.at(i);
        auto m = modelMatrices.at(i);
		Mesh newMesh = Mesh((std::filesystem::current_path() / string).string(), context, renderPipeline->getCommandList(), renderPipeline, m);
		meshes.push_back(newMesh);
		sceneSize += newMesh.getNumTriangles();
	}
}

void ObjectScene::draw(Camera* camera) {
    for (Mesh m : meshes) {
        // == IA ==
        auto cmdList = renderPipeline->getCommandList();
        cmdList->IASetVertexBuffers(0, 1, m.getVBV());
        cmdList->IASetIndexBuffer(m.getIBV());
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // == RS ==
        //NO NEED TO RESET VIEWPORT??
        // == PSO ==
        cmdList->SetPipelineState(renderPipeline->getPSO());
        cmdList->SetGraphicsRootSignature(renderPipeline->getRootSignature());
        // == ROOT ==
        auto viewMat = camera->getViewMat();
        auto projMat = camera->getProjMat();
        auto modelMat = XMLoadFloat4x4(m.getModelMatrix());
        XMMATRIX mvpMatrix = XMMatrixMultiply(modelMat, XMMatrixMultiply(viewMat, projMat));

        cmdList->SetGraphicsRoot32BitConstants(0, 16, &mvpMatrix, 0);
        
        // Inverse transpose of the model matrix
        XMMATRIX inverseMatrix = XMMatrixInverse(nullptr, modelMat);
        XMMATRIX inverseTransposeMatrix = XMMatrixTranspose(inverseMatrix);
        cmdList->SetGraphicsRoot32BitConstants(0, 16, &inverseTransposeMatrix, 16);

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
    renderPipeline->releaseResources();
}