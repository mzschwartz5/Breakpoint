#include "BilevelUniformGridRootSig.hlsl"
#include "../constants.h"

struct Cell {
    int particleCount;
    int particleIndices[MAX_PARTICLES_PER_CELL];
};

struct Block {
    int nonEmptyCellCount;
};

// SRV for positions buffer (input buffer)
StructuredBuffer<float3> positionsBuffer : register(t0);

// UAV for the bilevel uniform grid (output buffers)
RWStructuredBuffer<Cell> cells : register(u0);
RWStructuredBuffer<Block> blocks : register(u1);

cbuffer BilevelUniformGridConstants : register(b0) {
    uint3 dimensions;
    float3 minBounds;
    float resolution;
    float numParticles;
};

uint3 getCellIndex(float3 particlePosition) {
    int cellIdxX = floor(particlePosition.x - minBounds.x) / resolution;
    int cellIdxY = floor(particlePosition.y - minBounds.y) / resolution;
    int cellIdxZ = floor(particlePosition.z - minBounds.z) / resolution;

    return uint3(cellIdxX, cellIdxY, cellIdxZ);
}

[numthreads(BILEVEL_UNIFORM_GRID_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= numParticles) {
        return;
    }

    float3 position = positionsBuffer[globalThreadId.x];
    uint3 cellIndices = getCellIndex(position);
    int cellIndex1D = cellIndices.x + cellIndices.y * dimensions.x + cellIndices.z * dimensions.x * dimensions.y;

    uint3 blockIndices = cellIndices / CELLS_PER_BLOCK_EDGE;
    float3 blockCenterPos = float3(blockIndices.x, blockIndices.y, blockIndices.z) * resolution * CELLS_PER_BLOCK_EDGE 
                            + (resolution * CELLS_PER_BLOCK_EDGE / 2.0f);

    // (±1 or 0, ±1 or 0, ±1 or 0)
    int3 octant = int3(sign(position - blockCenterPos));
    // The case where sign(x) = 0 is both annoying to deal with and frankly inconsequential to the simulation
    // so just set to 1 in this case to avoid slow if-conditions in the nested loop below.
    octant = int3(octant.x == 0 ? 1 : octant.x, 
                  octant.y == 0 ? 1 : octant.y, 
                  octant.z == 0 ? 1 : octant.z);

    // Add this particle to the cell
    int particleIndexInCell;
    InterlockedAdd(cells[cellIndex1D].particleCount, 1, particleIndexInCell);
    if (particleIndexInCell < MAX_PARTICLES_PER_CELL) {
        cells[cellIndex1D].particleIndices[particleIndexInCell] = globalThreadId.x;
    }

    // Do this only once per cell
    if (particleIndexInCell == 0) {
        // Need to iterate over neighboring blocks and increment the nonEmptyCellCount
        for (uint i = 0; i <= 1; ++i) {
            for (uint j = 0; j <= 1; ++j) {
                for (uint k = 0; k <= 1; ++k) {
                    uint3 neighborBlockIndices = blockIndices + uint3(i, j, k) * octant;

                    if (neighborBlockIndices.x >= dimensions.x / CELLS_PER_BLOCK_EDGE ||
                        neighborBlockIndices.y >= dimensions.y / CELLS_PER_BLOCK_EDGE ||
                        neighborBlockIndices.z >= dimensions.z / CELLS_PER_BLOCK_EDGE) {
                        continue;
                    }

                    uint neighborBlockIndex1D = neighborBlockIndices.x 
                                            + neighborBlockIndices.y * dimensions.x / CELLS_PER_BLOCK_EDGE
                                            + neighborBlockIndices.z * dimensions.x / CELLS_PER_BLOCK_EDGE * dimensions.y / CELLS_PER_BLOCK_EDGE;
                    InterlockedAdd(blocks[neighborBlockIndex1D].nonEmptyCellCount, 1);
                }
            }
        }
    }
}