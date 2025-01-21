#include "DispatchArgDivideRootSig.hlsl"

struct ConstantBufferData {
    int threadGroupSize;
};

// Dispatch args (assumed to be 1D, despite having 3 components)
RWStructuredBuffer<int3> dispatchArgs : register(u0);

// Number to divide by (thread group size)
ConstantBuffer<ConstantBufferData> cb : register(b0);

[numthreads(1, 1, 1)]
void main() {
    // Division rounding up
    dispatchArgs[0].x = (dispatchArgs[0].x + cb.threadGroupSize - 1) / cb.threadGroupSize;
}