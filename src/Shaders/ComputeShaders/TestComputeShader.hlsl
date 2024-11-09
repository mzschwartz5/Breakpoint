#include "TestRootSignature.hlsl"  // Includes the ROOTSIG definition

// UAV for float3 buffer (output buffer)
RWStructuredBuffer<float3> outputBuffer : register(u0);

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
    uint id = DTid.x;

    // Read the float3 from the buffer, increment the y component, and write back
    float3 value = outputBuffer[id];
    value.y += 2.2f;
    outputBuffer[id] = value;
}