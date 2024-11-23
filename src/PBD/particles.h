#include <iostream>

#include "Support/WinInclude.h"
#include "Support/ComPointer.h"
#include "Support/Window.h"
#include "Support/Shader.h"

#include "Debug/DebugLayer.h"


#include "Scene/Camera.h"



struct Particle {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 prevPosition;
    DirectX::XMFLOAT3 velocity;
    float invMass;
};

struct Voxel {
    UINT particleIndices[8]; // Indices to the 8 corner particles
    XMFLOAT3 u; // Local X-axis
    XMFLOAT3 v; // Local Y-axis
    XMFLOAT3 w; // Local Z-axis
    bool faceConnections[6]; // Store connection state for each face (+X,-X,+Y,-Y,+Z,-Z)
    float faceStrains[6];

    XMFLOAT3 centroidVelocity;
    float accumulatedStrain;
};

struct FaceConstraint {
    UINT voxelA;
    UINT voxelB;
    UINT faceIndex; // Identifies which face is shared
};

//testing
struct DistanceConstraint {
    UINT particleA;
    UINT particleB;
    float restLength;

    
};




struct SimulationParams {
    UINT constraintCount;
    float deltaTime;
    float count;
    float breakingThreshold;
    float randomSeed;
    float strainMemory;
    float rotationalInertia;
    DirectX::XMFLOAT3 gravity;
};