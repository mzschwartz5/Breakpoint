#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=9, b0),"     /* 9 constants: gravityStrength, inputX, inputY, deltaTime */ \
"DescriptorTable(UAV(u0, numDescriptors=2))"  /* UAV table for positions and velocities buffers */
