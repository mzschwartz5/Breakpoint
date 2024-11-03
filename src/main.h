#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "Support/Shader.h"
#include "DebugLayer/DebugLayer.h"
#include "D3D/DXContext.h"

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

// === Vertex Data ===
struct Vertex
{
    float x, y;
};
Vertex verticies[] =
{
    // T1
    { -1.f, -1.f },
    {  0.f,  1.f },
    {  1.f, -1.f },
};
D3D12_INPUT_ELEMENT_DESC vertexLayout[] =
{
    { "Position", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};


