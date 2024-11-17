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

struct BilevelUniformGridConstants {
    uint3 dimensions;
    float3 minBounds;
    float resolution;
    int numParticles;
};

ConstantBuffer<BilevelUniformGridConstants> cb : register(b0);

uint3 getCellIndex(float3 particlePosition) {
    int cellIdxX = floor(particlePosition.x - cb.minBounds.x) / cb.resolution;
    int cellIdxY = floor(particlePosition.y - cb.minBounds.y) / cb.resolution;
    int cellIdxZ = floor(particlePosition.z - cb.minBounds.z) / cb.resolution;

    return uint3(cellIdxX, cellIdxY, cellIdxZ);
}

[numthreads(BILEVEL_UNIFORM_GRID_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID) {
    if (globalThreadId.x >= cb.numParticles) {
        return;
    }

    float3 position = positionsBuffer[globalThreadId.x];
    uint3 cellIndices = getCellIndex(position);
    int cellIndex1D = cellIndices.x + cellIndices.y * cb.dimensions.x + cellIndices.z * cb.dimensions.x * cb.dimensions.y;
    uint3 blockIndices = cellIndices / CELLS_PER_BLOCK_EDGE;
    uint3 localCellIndices = cellIndices - (blockIndices * CELLS_PER_BLOCK_EDGE); // could be done with modulo, but this is faster

    // (±1 or 0, ±1 or 0, ±1 or 0)
    // 1 indicates cell is on a positive edge of a block, -1 indicates cell is on a negative edge, 0 indicates cell is not on an edge
    // This should work for both even and odd CELLS_PER_BLOCK_EDGE
    float halfCellsPerBlockEdge = (CELLS_PER_BLOCK_EDGE / 2.0);
    int3 edge = int3(trunc((localCellIndices - halfCellsPerBlockEdge) / halfCellsPerBlockEdge));

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
        for (uint i = 0; i <= abs(edge.x); ++i) {
            for (uint j = 0; j <= abs(edge.y); ++j) {
                for (uint k = 0; k <= abs(edge.z); ++k) {
                    uint3 neighborBlockIndices = blockIndices + uint3(i, j, k) * edge;

                    if (neighborBlockIndices.x >= cb.dimensions.x / CELLS_PER_BLOCK_EDGE ||
                        neighborBlockIndices.y >= cb.dimensions.y / CELLS_PER_BLOCK_EDGE ||
                        neighborBlockIndices.z >= cb.dimensions.z / CELLS_PER_BLOCK_EDGE) {
                        continue;
                    }

                    uint neighborBlockIndex1D = neighborBlockIndices.x 
                                            + neighborBlockIndices.y * cb.dimensions.x / CELLS_PER_BLOCK_EDGE
                                            + neighborBlockIndices.z * cb.dimensions.x / CELLS_PER_BLOCK_EDGE * cb.dimensions.y / CELLS_PER_BLOCK_EDGE;
                    InterlockedAdd(blocks[neighborBlockIndex1D].nonEmptyCellCount, 1);
                }
            }
        }
    }
}