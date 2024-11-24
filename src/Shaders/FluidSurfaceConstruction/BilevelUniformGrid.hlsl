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

ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

uint3 getCellIndex(float3 particlePosition) {
    int cellIdxX = floor((particlePosition.x - cb.minBounds.x) / cb.resolution);
    int cellIdxY = floor((particlePosition.y - cb.minBounds.y) / cb.resolution);
    int cellIdxZ = floor((particlePosition.z - cb.minBounds.z) / cb.resolution);

    return uint3(cellIdxX, cellIdxY, cellIdxZ);
}

// NOTE: if this compute shader changes to 3D, the logic also needs to change to get and use the particle index correctly.
[numthreads(BILEVEL_UNIFORM_GRID_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= cb.numParticles) {
        return;
    }

    float3 position = positionsBuffer[globalThreadId.x];
    int3 cellIndices = getCellIndex(position);
    int cellIndex1D = cellIndices.x + (cellIndices.y * cb.dimensions.x) + (cellIndices.z * cb.dimensions.x * cb.dimensions.y);
    int3 blockIndices = cellIndices / CELLS_PER_BLOCK_EDGE;
    int3 localCellIndices = cellIndices - (blockIndices * CELLS_PER_BLOCK_EDGE); // could be done with modulo, but this is faster since we already have blockIndices

    // (±1 or 0, ±1 or 0, ±1 or 0)
    // 1 indicates cell is on a positive edge of a block, -1 indicates cell is on a negative edge, 0 indicates cell is not on an edge
    // This should work for both even and odd CELLS_PER_BLOCK_EDGE
    float halfCellsPerBlockEdgeMinusOne = ((CELLS_PER_BLOCK_EDGE - 1) / 2.0);
    int3 edge = int3(trunc((localCellIndices - halfCellsPerBlockEdgeMinusOne) / halfCellsPerBlockEdgeMinusOne));

    // Add this particle to the cell
    int particleIndexInCell;
    InterlockedAdd(cells[cellIndex1D].particleCount, 1, particleIndexInCell);
    if (particleIndexInCell < MAX_PARTICLES_PER_CELL) {
        cells[cellIndex1D].particleIndices[particleIndexInCell] = globalThreadId.x;
    }

    // Do this only once per cell, when the first particle is added to the cell
    if (particleIndexInCell == 0) {
        // Increment the nonEmptyCellCount of the block, and any blocks for which this cell borders on (which can be a maximum of 8)
        // `edge` is carefully calculated so that these nested loops will exactly iterate over each abutting neighbor block.
        int3 gridBlockDimensions = cb.dimensions / CELLS_PER_BLOCK_EDGE;
        for (int i = 0; i <= abs(edge.x); ++i) {
            for (int j = 0; j <= abs(edge.y); ++j) {
                for (int k = 0; k <= abs(edge.z); ++k) {
                    int3 neighborBlockIndices = blockIndices + (int3(i, j, k) * edge);

                    // TODO: can avoid checking this every loop by using min/max clamps
                    // and looping over global indices (see SurfaceCellDetection.hlsl)
                    if (any(neighborBlockIndices >= gridBlockDimensions) || 
                        any(neighborBlockIndices < 0)) {
                        continue;
                    }

                    uint neighborBlockIndex1D = neighborBlockIndices.x 
                                              + neighborBlockIndices.y * gridBlockDimensions.x
                                              + neighborBlockIndices.z * gridBlockDimensions.x * gridBlockDimensions.y;

                    // TODO: test if its faster to atomic add first to a shared memory variable, then add to the buffer at the end
                    InterlockedAdd(blocks[neighborBlockIndex1D].nonEmptyCellCount, 1);
                }
            }
        }
    }
}