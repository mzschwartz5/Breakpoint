#pragma once

static const float PI = 3.1415926f;

/* ================== Constants for the mesh shading pipeline ================== */ 
// This is just for shaders, types will not compile in C++ code
#ifndef __cplusplus
struct BilevelUniformGridConstants {
    int numParticles;
    uint3 dimensions;
    float3 minBounds;
    float resolution;
};
#endif

static const int KERNEL_SCALE = 15; // not exactly sure what the significance of this is, copied from paper's repo. (it can vary between 0.05 and 20.0f via a UI in the repo)
static const int MAX_PARTICLES_PER_CELL = 8;
// Generally keep this to a power of two
static const int CELLS_PER_BLOCK_EDGE = 4; // each block has 4x4x4 cells
static const int CELLS_PER_BLOCK = CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE;
static const int FILLED_BLOCK = (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2);
static const int BILEVEL_UNIFORM_GRID_THREADS_X = 64;
static const int SURFACE_BLOCK_DETECTION_THREADS_X = 64;
// For this compute pass, its important for the workgroup size to match the number of cells per block,
// because we use shared memory in this pass and its most coherent if workgroups map to blocks.
static const int SURFACE_CELL_DETECTION_THREADS_X = CELLS_PER_BLOCK;
// For this compute pass as well, it's important that the workgroup size matches the number of vertices per block.
static const int SURFACE_VERTEX_COMPACTION_THREADS_X = (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1) * (CELLS_PER_BLOCK_EDGE + 1);
// It's actually not so important for the workgroup size for this pass to match the number of vertices per block. By this point
// we no longer have block-level coherency of vertices anyway. But use this as a starting point - can adjust it later.
static const int SURFACE_VERTEX_DENSITY_THREADS_X = SURFACE_VERTEX_COMPACTION_THREADS_X;