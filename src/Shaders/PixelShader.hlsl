#include "RootSignature.hlsl"

struct PSInput {
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
};

[RootSignature(ROOTSIG)]
void main(PSInput input, out float4 color : SV_Target0, out float4 position : SV_Target1) : SV_Target
{
    float4 baseColor = float4(0.8, 0.7, 0.6, 1);

    // Basic lambertian shading
    float3 lightDir = normalize(float3(1, 1, 1));
    float3 normal = normalize(input.normal.xyz);
    float intensity = saturate(dot(normal, lightDir));
    float4 ambientLight = float4(0.2, 0.2, 0.2, 1);
    color =  ambientLight + baseColor * float4(intensity, intensity, intensity, 1);
    position = input.position;
}