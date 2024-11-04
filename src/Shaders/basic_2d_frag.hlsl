#include "root_signature.hlsl"
[RootSignature(ROOTSIG)]

float4 main() : SV_Target
{
	return float4(1.0, 0.5, 0.5, 1.0f);
}