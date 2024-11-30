#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=18, b0),"     /* 17 constants: w/ one int2, see PBMPM Constants */ \
"DescriptorTable(UAV(u0, numDescriptors=3)),"    /* Table for particleBuffer, freeIndicesBuffer, particleCountBuffer */ \
"DescriptorTable(SRV(t0, numDescriptors=1))," /* Table for SimShapes*/ \
"DescriptorTable(SRV(t1, numDescriptors=1))" /* Table for curr grid */ \

