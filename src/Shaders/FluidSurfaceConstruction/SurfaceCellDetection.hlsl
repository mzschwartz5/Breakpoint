#include "SurfaceCellDetectionRootSig.hlsl"
#include "../constants.h"
#include "utils.hlsl"

// TODO: should this be a constant? If not, what happens if we change it from frame to frame? Do we have to resize and reset buffers as well?
struct Uniforms {
    uint3 gridCellDimensions;
};

struct Cell {
    int particleCount;
    int particleIndices[MAX_PARTICLES_PER_CELL];
};

// Inputs
// Constant buffer (root constant)
ConstantBuffer<Uniforms> cb : register(b0);
// SRV for surface block indices
StructuredBuffer<uint> surfaceBlockIndices : register(t0);
// SRV for the surface cells
StructuredBuffer<Cell> cells : register(t1);
// SRV for the surface block dispatch
StructuredBuffer<uint3> surfaceBlockDispatch : register(t2);

// Outputs
RWStructuredBuffer<uint> surfaceVertices : register(u0);

// At a typical value of FILLED_BLOCK = 216, (4 + 2)^3, this is well within the limits of shared memory. 
groupshared uint cellParticleCounts[FILLED_BLOCK];

// Since surface cells share vertices, keep track of whether the verts are surface verts in shared memory first,
// then write them to the output buffer. This avoids excess global memory writes.
groupshared uint s_surfaceVertices[(CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1)];

// Set the 8 vertices of a surface cell cube into shared memory
void setSharedMemSurfaceVerts(uint3 baseIndex3d, uint3 dimensions, uint value) {
    s_surfaceVertices[to1D(baseIndex3d + uint3(0, 0, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(0, 0, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(0, 1, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(0, 1, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(1, 0, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(1, 0, 1), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(1, 1, 0), dimensions)] = value;
    s_surfaceVertices[to1D(baseIndex3d + uint3(1, 1, 1), dimensions)] = value;
}

// Set the 8 vertices of a surface cell cube into the output buffer
void setGlobalSurfaceVertsToValue(uint3 globalIndex3d, uint3 globalDims, uint value) {
    surfaceVertices[to1D(globalIndex3d + uint3(0, 0, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(0, 0, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(0, 1, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(0, 1, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(1, 0, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(1, 0, 1), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(1, 1, 0), globalDims)] = value;
    surfaceVertices[to1D(globalIndex3d + uint3(1, 1, 1), globalDims)] = value;
}

// Set the 8 vertices of a surface cell cube into the output buffer, from shared memory.
void setGlobalSurfaceVertsFromSharedMem(uint3 globalIndex3d, uint3 globalDims, uint3 sharedIndex3d, uint3 sharedDims) {
    surfaceVertices[to1D(globalIndex3d + uint3(0, 0, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(0, 0, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(0, 0, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(0, 0, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(0, 1, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(0, 1, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(0, 1, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(0, 1, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(1, 0, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(1, 0, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(1, 0, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(1, 0, 1), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(1, 1, 0), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(1, 1, 0), sharedDims)];
    surfaceVertices[to1D(globalIndex3d + uint3(1, 1, 1), globalDims)] = s_surfaceVertices[to1D(sharedIndex3d + uint3(1, 1, 1), sharedDims)];
}

void setVertSharedMemory(uint index, uint value) {
    s_surfaceVertices[index] = value;
}

void setVertGlobalMemory(uint index, uint value) {
    surfaceVertices[index] = value;
}

// NOTE: the logic in this shader RELIES on the number of threads per workgroup equalling the number in a single block. 
[numthreads(SURFACE_CELL_DETECTION_THREADS_X, 1, 1)]
void main(uint3 globalThreadId : SV_DispatchThreadID, uint3 localThreadId : SV_GroupThreadID) {
    if (globalThreadId.x >= surfaceBlockDispatch[0].x * CELLS_PER_BLOCK) {
        return;
    }

    // First order of business: we launched a thread for each cell within surface blocks. We need to figure out the global index for this thread's cell. 
    uint surfaceBlockIdx1d = surfaceBlockIndices[globalThreadId.x / CELLS_PER_BLOCK];
    // By aligning the number of threads per workgroup with the number of threads in a block, we can use the local thread ID as a proxy for the local cell index.
    uint3 surfaceBlockIdx3d = to3D(surfaceBlockIdx1d, cb.gridCellDimensions / CELLS_PER_BLOCK_EDGE);
    uint3 localCellIndex3d = to3D(localThreadId.x, CELLS_PER_BLOCK_EDGE * uint3(1, 1, 1));
    uint3 globalCellIndex3d = surfaceBlockIdx3d * CELLS_PER_BLOCK + localCellIndex3d;

    // A quick aside: using the globalCellIndex3d, reset the surface vertices buffer (from last frame) to 0
    setGlobalSurfaceVertsToValue(globalCellIndex3d, cb.gridCellDimensions + 1, 0);

    // Prefetch and store the particle count for this cell. Since shared memory will store (N + 2)^3 cells, not just N^3 cells, in order for all cells to be contiguous,
    // we need to offset the surface block cells themselves by 1 to leave room for the extra-surface cells.
    // (Skip the cells on edges, we'll get them later)
    uint offsetLocalCellIdx1d = to1D(localCellIndex3d + uint3(1, 1, 1), CELLS_PER_BLOCK_EDGE + 2);
    if (all(localCellIndex3d < (CELLS_PER_BLOCK_EDGE - 1) * uint3(1, 1, 1)) && all(localCellIndex3d > uint3(0, 0, 0))) {
        cellParticleCounts[offsetLocalCellIdx1d] = cells[globalThreadId.x].particleCount;
    }

    // We also need the particle counts for the cells surrounding the surface block. Let each edge cell in the surface block
    // fetch its surrounding extra-surface cells (and itself).
    // This follows the same strategy as the bilevel uniform grid shader.
    float halfCellsPerBlockEdgeMinusOne = ((CELLS_PER_BLOCK_EDGE - 1) / 2.0);
    uint3 edge = int3(trunc((localCellIndex3d - halfCellsPerBlockEdgeMinusOne) / halfCellsPerBlockEdgeMinusOne));
    // By converting neighbors to global-space here, we can clamp to the grid bounds and avoid 
    // out-of-bounds checks in every loop iteration.
    uint3 globalNeighborCells = clamp(globalCellIndex3d + edge, uint3(0, 0, 0), cb.gridCellDimensions - 1);
    uint3 minSearchBounds = min(globalNeighborCells, globalCellIndex3d);
    uint3 maxSearchBounds = max(globalNeighborCells, globalCellIndex3d);

    uint3 globalNeighborCellOrigin = surfaceBlockIdx3d * CELLS_PER_BLOCK - uint3(1, 1, 1);
    for (uint z = minSearchBounds.z; z <= maxSearchBounds.z; z++) {
        for (uint y = minSearchBounds.y; y <= maxSearchBounds.y; y++) {
            for (uint x = minSearchBounds.x; x <= maxSearchBounds.x; x++) {
                uint3 globalNeighborCellIndex3d = uint3(x, y, z);
                uint globalNeighborCellIndex1d = to1D(globalNeighborCellIndex3d, cb.gridCellDimensions);
                
                uint3 localNeighborCellIndex3d = globalNeighborCellIndex3d - globalNeighborCellOrigin;
                uint localNeighborCellIndex1d = to1D(localNeighborCellIndex3d, CELLS_PER_BLOCK_EDGE + 2);
                
                cellParticleCounts[localNeighborCellIndex1d] = cells[globalNeighborCellIndex1d].particleCount;
            }
        }
    }

    // Now that all the memory we'll need is pulled into shared, sync it, and then process it to detect surface cells.
    GroupMemoryBarrierWithGroupSync();

    // Similar trick as before to get the search bounds, but this time we're iterating over the full 3x3x3 neighboring cells (within grid bounds).
    minSearchBounds = clamp(globalCellIndex3d - uint3(1, 1, 1), uint3(0, 0, 0), cb.gridCellDimensions - 1);
    maxSearchBounds = clamp(globalCellIndex3d + uint3(1, 1, 1), uint3(0, 0, 0), cb.gridCellDimensions - 1);

    // A cell is NOT a surface cell if either:
    // 1. All of its neighbors and itself are empty
    // 2. All of its neighbors and itself are filled
    // It CAN be a surface cell EVEN if it has no particles itself.
    // As we iterate over all neighbors, if any one of them is different from the previous ones, the cell is a surface cell.
    // To track this, initialize lastEmptyCell to the state of the current cell, from shared memory.
    bool lastCellEmpty = (cellParticleCounts[offsetLocalCellIdx1d] == 0);
    bool isSurfaceCell = false;
    for (uint z = minSearchBounds.z; z <= maxSearchBounds.z; z++) {
        for (uint y = minSearchBounds.y; y <= maxSearchBounds.y; y++) {
            for (uint x = minSearchBounds.x; x <= maxSearchBounds.x; x++) {
                uint3 globalNeighborCellIndex3d = uint3(x, y, z);

                uint3 localNeighborCellIndex3d = globalNeighborCellIndex3d - globalNeighborCellOrigin;
                uint localNeighborCellIndex1d = to1D(localNeighborCellIndex3d, CELLS_PER_BLOCK_EDGE + 2);

                bool isCellEmpty = (cellParticleCounts[localNeighborCellIndex1d] == 0);
                if (isCellEmpty != lastCellEmpty) {
                    isSurfaceCell = true;
                    break;
                };
            }
            if (isSurfaceCell) {
                break;
            }
        }
        if (isSurfaceCell) {
            break;
        }
    }

    // Initialize the shared memory to all 0s, then only write to it if isSurfaceCell.
    // We write to shared memory first because cells can share vertices, so this saves on duplicate global writes.
    setSharedMemSurfaceVerts(localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1), 0);
    GroupMemoryBarrierWithGroupSync();

    if (isSurfaceCell) {
        setSharedMemSurfaceVerts(localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1) * uint3(1, 1, 1), 1);
    }
    GroupMemoryBarrierWithGroupSync();

    // Set surface vertices buffer
    if (isSurfaceCell) {
        setGlobalSurfaceVertsFromSharedMem(globalCellIndex3d, (cb.gridCellDimensions + 1), localCellIndex3d, (CELLS_PER_BLOCK_EDGE + 1));
    }
}