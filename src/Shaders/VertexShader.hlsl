#include "RootSignature.hlsl"

cbuffer CameraMatrices : register(b0) {
    float4x4 viewMatrix;        // 16 floats
    float4x4 projectionMatrix;  // 16 floats
};

[RootSignature(ROOTSIG)]
float4 main(float3 pos : Position) : SV_Position
{
    //return mul(mul(float4(pos, 1.0f), viewMatrix), projectionMatrix);
    return mul(projectionMatrix, mul(viewMatrix, float4(pos, 1.0f)));
}