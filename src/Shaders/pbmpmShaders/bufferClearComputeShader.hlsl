#include "bufferClearRootSignature.hlsl"  // Includes the ROOTSIG definition

// Root constants bound to b0
cbuffer bufferSize : register(b0)
{
    uint size;
};
// UAVs and SRVs
RWStructuredBuffer<int> buffer : register(u0);

// Compute Shader Entry Point
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    // Ensure the current invocation is within bounds
    if (id.x >= size)
    {
        return;
    }
    
    buffer[id.x] = 0;
}