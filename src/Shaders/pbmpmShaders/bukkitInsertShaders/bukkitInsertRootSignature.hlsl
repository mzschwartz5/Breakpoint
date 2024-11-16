#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"CBV(b1),"     /* Root CBV for g_particleCount */ \
"DescriptorTable(SRV(t0, numDescriptors=1)),"    /* SRVs for g_particles, g_bukkitIndexStart */ \
"DescriptorTable(SRV(t1, numDescriptors=1))," \
"DescriptorTable(UAV(u0, numDescriptors=1))"  \   /* UAVs for g_particleInsertCounters and g_particleData */
"DescriptorTable(UAV(u1, numDescriptors=1))"