#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"DescriptorTable(UAV(u0, numDescriptors=2), CBV(b1, numDescriptors=2)"    /* Table for particleBuffer, freeIndicesBuffer, particleCountBuffer, and simDispatchBuffer*/ \
"DescriptorTable(UAV(u1, numDescriptors=2), SRV(t0, numDescriptors=2), CBV(b2, numDescriptors=2))," /* Table for BukkitSystemBuffers */ \
"DescriptorTable(SRV(t2, numDescriptors=1), UAV(u8, numDescriptors=2)," /* Table for three grid buffers */

