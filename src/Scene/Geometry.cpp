#include "Geometry.h"

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

    return { vertices, indices };
}


std::pair<std::vector<XMFLOAT3>, std::vector<unsigned int>> generateSphere(float radius, int sliceCount, int stackCount) {
    std::vector<XMFLOAT3> vertices;
    std::vector<unsigned int> indices;

    // Generate vertices
    for (int i = 0; i <= stackCount; ++i) {
        float phi = XM_PI * i / stackCount; // From 0 to PI

        for (int j = 0; j <= sliceCount; ++j) {
            float theta = 2.0f * XM_PI * j / sliceCount; // From 0 to 2PI

            // Spherical to Cartesian coordinates
            float x = radius * sinf(phi) * cosf(theta);
            float y = radius * cosf(phi);
            float z = radius * sinf(phi) * sinf(theta);

            XMFLOAT3 position = XMFLOAT3(x, y, z);

            vertices.push_back( position);
        }
    }

    // Generate indices
    for (int i = 0; i < stackCount; ++i) {
        for (int j = 0; j < sliceCount; ++j) {
            int first = i * (sliceCount + 1) + j;
            int second = first + sliceCount + 1;

            // First triangle of quad
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            // Second triangle of quad
            indices.push_back(first + 1);
            indices.push_back(second);
            indices.push_back(second + 1);
        }
    }

    return { vertices, indices };
}