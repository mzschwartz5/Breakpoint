#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0)," \   /* 17 constants: w/ one int2, see PBMPM Constants */
"DescriptorTable(SRV(t0, numDescriptors=1))," \    /* SRV table for g_gridSrc, g_bukkitThreadData, g_bukkitParticleData */
"DescriptorTable(SRV(t1, numDescriptors=1))," \
"DescriptorTable(SRV(t2, numDescriptors=1))," \
"DescriptorTable(UAV(u0, numDescriptors=1))" \   /* UAV table for g_particles, g_gridDst, g_gridToBeCleared, g_freeIndices */
"DescriptorTable(UAV(u1, numDescriptors=1))" \
"DescriptorTable(UAV(u2, numDescriptors=1))" \
"DescriptorTable(UAV(u3, numDescriptors=1))"

