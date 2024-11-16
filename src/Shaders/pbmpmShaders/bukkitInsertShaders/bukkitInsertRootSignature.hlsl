#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"CBV(b1),"     /* Root CBV for g_particleCount */ \
"DescriptorTable(SRV(t0, numDescriptors=2)),"    /* SRVs for g_particles, g_bukkitIndexStart */ \
"DescriptorTable(UAV(u0, numDescriptors=2))"     /* UAVs for g_particleInsertCounters and g_particleData */