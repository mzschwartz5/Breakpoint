#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Support/WinInclude.h"

#include "D3D/DXContext.h"
#include "D3D/VertexBuffer.h"
#include "D3D/IndexBuffer.h"
#include "D3D/ModelMatrixBuffer.h"

using namespace DirectX;

struct Vertex {
	XMFLOAT3 pos;
	XMFLOAT3 nor;
	XMFLOAT3 col;
};

class Mesh {
public:
	Mesh() = delete;
	Mesh(std::string fileLocation, DXContext* context, ID3D12GraphicsCommandList5* cmdList, RenderPipeline* pipeline);
	void loadMesh(std::string fileLocation);

	D3D12_INDEX_BUFFER_VIEW* getIBV();
	D3D12_VERTEX_BUFFER_VIEW* getVBV();
	ModelMatrixBuffer* getMMB();

	void releaseResources();

	size_t getNumTriangles();

private:
	std::vector<Vertex> vertices;
	std::vector<XMFLOAT3> vertexPositions;
	std::vector<unsigned int> indices;
	std::vector<XMFLOAT4X4> modelMatrices;
	int numInstances;
	size_t numTriangles;

	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;
	ModelMatrixBuffer modelMatrixBuffer;
	D3D12_VERTEX_BUFFER_VIEW vbv;
	D3D12_INDEX_BUFFER_VIEW ibv;
};