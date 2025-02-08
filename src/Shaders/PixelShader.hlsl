#include "RootSignature.hlsl"

struct PSInput {
    float4 position : SV_POSITION;
    float4 normal : NORMAL;
    float4 worldPos : TEXCOORD0;
};

[RootSignature(ROOTSIG)]
/**
* This pixel shader is used to render to three render targets.
* The first render target is used to store the color of the object.
* The second render target is used to store the position of the object.
* The third render target is used to store the color of the main render target.
*/
void main(PSInput input, out float4 color : SV_Target0, out float4 position : SV_Target1, out float4 mainRTColor : SV_TARGET2) : SV_Target
{
    float4 baseColor = float4(0.8, 0.7, 0.6, 1);

    // Basic lambertian shading
    float3 lightDir = normalize(float3(1, 1, 1));
    float3 normal = normalize(input.normal.xyz);
    float intensity = saturate(dot(normal, lightDir));
    float4 ambientLight = float4(0.2, 0.2, 0.2, 1);
    color = ambientLight + baseColor * float4(intensity, intensity, intensity, 1);
    position = input.worldPos;
    mainRTColor = color;
}