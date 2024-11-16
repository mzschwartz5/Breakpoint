#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"UAV(u0),"                                      /* UAV for g_bukkitIndirectDispatch */ \
"UAV(u1),"                                      /* UAV for g_bukkitParticleAllocator*/ \
"DescriptorTable(SRV(t0)),"                      /* SRV for g_bukkitCounts */ \
"DescriptorTable(UAV(u2, numDescriptors=2))"     /* UAVs for g_bukkitThreadData, g_bukkitIndexStart */