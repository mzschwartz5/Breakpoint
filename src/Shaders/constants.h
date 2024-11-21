#pragma once

// Constants for the mesh shading pipeline
static const int MAX_PARTICLES_PER_CELL = 8;
// Generally keep this to a power of two
static const int CELLS_PER_BLOCK_EDGE = 4; // each block has 4x4x4 cells
static const int CELLS_PER_BLOCK = CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE * CELLS_PER_BLOCK_EDGE;
static const int FILLED_BLOCK = (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2) * (CELLS_PER_BLOCK_EDGE + 2);
static const int BILEVEL_UNIFORM_GRID_THREADS_X = 32;
static const int SURFACE_BLOCK_DETECTION_THREADS_X = 32;
// For this compute pass, its important for the workgroup size to match the number of cells per block,
// because we use shared memory in this pass and its most coherent if workgroups map to blocks.
static const int SURFACE_CELL_DETECTION_THREADS_X = CELLS_PER_BLOCK;

