#include "RootSignature.hlsl"

[RootSignature(ROOTSIG)]
float4 main() : SV_Target
{
    return float4(1.0, 0.25, 0.4, 1.0f);
}