#include "RootSignature.hlsl"

[RootSignature(ROOTSIG)]
float4 main() : SV_Target
{
    return float4(0.00, 0.63, 0.98, 1.0f);
}