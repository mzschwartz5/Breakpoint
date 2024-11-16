#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"CBV(b1),"     /* Root CBV for g_particleCount */ \
"DescriptorTable(SRV(t0, numDescriptors=1)),"     /* SRV table for g_particles */ \
"DescriptorTable(UAV(u0, numDescriptors=1))"      /* UAV table for g_bukkitCounts */