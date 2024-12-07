#include "PBMPMVertexRootSignature.hlsl"

cbuffer CameraMatrices : register(b0) {
    float4x4 viewMatrix;        // 16 floats
    float4x4 projectionMatrix;  // 16 floats
	float4x4 modelMatrix;       // 16 floats
};

// Particle positions as an SRV at register t0
StructuredBuffer<float4> positions : register(t0);

struct VSInput
{
    float3 Position : POSITION;      // Input position from vertex buffer
    uint InstanceID : SV_InstanceID; // Instance ID for indexing into model matrices
};

[RootSignature(ROOTSIG)]
float4 main(VSInput input) : SV_Position
{
    // Retrieve the particle position for the current instance
    float3 particlePosition = positions[input.InstanceID].xyz;

    // Apply the model, view, and projection transformations
    float4 worldPos = mul(modelMatrix, float4(input.Position + particlePosition, 1.0));
    float4 viewPos = mul(viewMatrix, worldPos);
    return mul(projectionMatrix, viewPos);
}