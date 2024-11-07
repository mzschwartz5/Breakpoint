#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "Support/Shader.h"

#include "Debug/DebugLayer.h"

#include "D3D/DXContext.h"
#include "D3D/RenderPipelineHelper.h"
#include "D3D/RenderPipeline.h"
#include "D3D/VertexBuffer.h"
#include "D3D/IndexBuffer.h"
#include "D3D/ModelMatrixBuffer.h"

#include "Scene/Camera.h"



struct Particle {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 prevPosition;
    DirectX::XMFLOAT3 velocity;
    float invMass;
};

struct Voxel {
    Particle* particles[8];
    // Additional data if needed
};

struct GPU_Constraint {
    UINT type;         // Constraint type (0: Orthogonality, 1: Length, 2: Glue)
    UINT indices[4];    
    float stiffness;
    float breakingThreshold;
    UINT isBroken;
};