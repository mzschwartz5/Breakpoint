#pragma once

#include "DirectXMath.h"

using namespace DirectX;

// Right triangle vertex data
std::vector<XMFLOAT3> rightTriVertices = {
    XMFLOAT3(-0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.25f, 0.25f, 0.0f)
};

// Equilateral triangle vertex data
std::vector<XMFLOAT3> equalTriVertices = {
    XMFLOAT3(-0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.0f, 0.1833f, 0.0f)
};

// Square vertex data
std::vector<XMFLOAT3> squareVertices = {
    XMFLOAT3(-0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.25f, -0.25f, 0.0f),
    XMFLOAT3(0.25f, 0.25f, 0.0f),
    XMFLOAT3(-0.25f, 0.25f, 0.0f)
};

// Index data
std::vector<unsigned int> triIndices = { 0, 1, 2 };
std::vector<unsigned int> squareIndices = { 0, 1, 2, 0, 2, 3 };

std::pair<std::vector<XMFLOAT3>, std::vector<unsigned int>> generateCircle(float radius, int segmentCount) {
    std::vector<XMFLOAT3> vertices;
    std::vector<unsigned int> indices;

    // Center vertex of the triangle fan
    vertices.push_back({ XMFLOAT3(0.0f, 0.0f, 0.0f) });

    // Generate vertices along the circumference
    for (int i = 0; i <= segmentCount; ++i) {
        float theta = 2.0f * XM_PI * i / segmentCount; // Angle for each segment
        float x = radius * cosf(theta);
        float y = radius * sinf(theta);
        vertices.push_back({ XMFLOAT3(x, y, 0.0f) });
    }

    // Generate indices for the triangle fan
    for (int i = 1; i <= segmentCount; ++i) {
        indices.push_back(0);       // Center vertex
        indices.push_back(i);       // Current vertex on the circumference
        indices.push_back(i + 1);   // Next vertex on the circumference
    }

    return {vertices, indices};
}