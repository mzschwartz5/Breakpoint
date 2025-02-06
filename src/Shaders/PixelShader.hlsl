#include "RootSignature.hlsl"

struct PSInput {
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
};

[RootSignature(ROOTSIG)]
float4 main(PSInput input) : SV_Target
{
    return input.normal;
}