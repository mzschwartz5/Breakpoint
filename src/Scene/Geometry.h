#pragma once

#include <vector>
#include "DirectXMath.h"

using namespace DirectX;

// Function declaration (not definition)
std::pair<std::vector<DirectX::XMFLOAT3>, std::vector<unsigned int>> generateCircle(float radius, int segments);
std::pair<std::vector<XMFLOAT3>, std::vector<unsigned int>> generateSphere(float radius, int sliceCount, int stackCount);

// External declarations for variables (do not initialize here)
extern std::vector<DirectX::XMFLOAT3> rightTriVertices;
extern std::vector<DirectX::XMFLOAT3> equalTriVertices;
extern std::vector<DirectX::XMFLOAT3> squareVertices;
extern std::vector<unsigned int> triIndices;
extern std::vector<unsigned int> squareIndices;