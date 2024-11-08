

#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=32, b0),"   /* For view and projection matrices */ \
"DescriptorTable(SRV(t0, space=0))"          /* SRV for model matrices buffer */ \
"DescriptorTable(SRV(t0, space=1))"          /* SRV for voxel grid */

#include "RootSignature.hlsl"

Texture3D<uint> voxelGrid : register(t0, space1);

struct PSInput {
    float4 Position : SV_POSITION;
};

[RootSignature(ROOTSIG)]
float4 main(PSInput input) : SV_Target{
   
    return float4(1.0, 0.25, 0.4, 1.0f);
}

