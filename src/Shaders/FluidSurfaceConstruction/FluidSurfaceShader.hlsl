#include "FluidMeshRootSig.hlsl"
#include "../constants.h"

struct PSInput {
    float4 ndcPos: SV_Position;
    float3 normal: NORMAL0;
    float3 worldPos: POSITION1;
#ifdef OUTPUT_MESHLETS
    int meshletIndex: COLOR0;
#endif
};

ConstantBuffer<MeshShadingConstants> cb : register(b0);

float3 radiance(float3 dir)
{
    // Paper uses a sky model for the radiance
    // Return a constant sky-like color
    return float3(0.53, 0.81, 0.92); // Light sky blue
}

float fresnelSchlick(float VdotH, float F0) {
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

float remapTo01(float value, float minValue, float maxValue) {
    return (value - minValue) / (maxValue - minValue);
}

float3 gammaCorrect(float3 color)
{
    float correctionFactor = (1.0 / 2.2);
    return pow(color, float3(correctionFactor, correctionFactor, correctionFactor));
}

float3 getMeshletColor(int index)
{
    float r = frac(sin(float(index) * 12.9898) * 43758.5453);
    float g = frac(sin(float(index) * 78.233) * 43758.5453);
    float b = frac(sin(float(index) * 43.853) * 43758.5453);
    return float3(r, g, b);
}

// Return the intersection point of a ray with an XZ plane at Y = 0
float3 planeRayIntersect(float3 origin, float3 direction)
{
    return origin - direction * (origin.y / direction.y);
}

static const float3 baseColor = float3(0.7, 0.9, 1);

[RootSignature(ROOTSIG)]
float4 main(PSInput input) : SV_Target
{
#ifdef OUTPUT_MESHLETS
    return float4(getMeshletColor(input.meshletIndex), 1.0);
#endif

    // refract
    float3 pos = input.worldPos;
    float3 dir = normalize(pos - cb.cameraPos);

    float ior = 1.33;
    float eta = 1.0 / ior;

    float3 reflectDir = reflect(dir, input.normal);
    float3 h = normalize(-dir + reflectDir);
    float3 reflection = radiance(reflectDir);
    float f0 = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
    float fr = fresnelSchlick(dot(-dir, h), f0);

    float3 refractDir = refract(dir, input.normal, eta);
    float3 refraction;

    float3 meshPos = planeRayIntersect(cb.cameraPos, dir);
    float dist = distance(pos, meshPos);
    float3 trans = clamp(exp(-remapTo01(dist, 1.0, 45.0)), 0.0, 1.0) * baseColor;
    refraction = trans * float4(0.7, 0.7, 0.85, 1.0);

    float3 baseColor = refraction * (1.0 - fr) + reflection * fr;
    return float4(gammaCorrect(baseColor), 1.0);
}