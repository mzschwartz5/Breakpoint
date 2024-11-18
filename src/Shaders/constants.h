#pragma once

// Constants for the mesh shading pipeline
static const int MAX_PARTICLES_PER_CELL = 8;
static const int BILEVEL_UNIFORM_GRID_THREADS_X = 32;
static const int CELLS_PER_BLOCK_EDGE = 4; // each block has 4x4x4 cells