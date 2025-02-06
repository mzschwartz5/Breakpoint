#include "RootSignature.hlsl"

cbuffer CameraMatrices : register(b0) {
    float4x4 mvpMatrix;     // 16 floats (model-view-projection matrix)
    float4x4 normalMatrix;  // 16 floats (inverse transpose of modelMatrix)
};

struct VSInput
{
    float4 position : POSITION;      // Input position from vertex buffer
    float4 normal : NORMAL;          // Input normal from vertex buffer
};

struct VSOutput
{
    float4 pos : SV_POSITION;
    float4 nor : NORMAL;
};

[RootSignature(ROOTSIG)]
VSOutput main(VSInput input)
{
    VSOutput output;
    output.pos = mul(mvpMatrix, input.position);
    output.nor = mul(normalMatrix, input.normal);
    return output;
}
