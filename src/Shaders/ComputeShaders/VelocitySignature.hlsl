#define ROOTSIG \
	"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
	"RootConstants(num32BitConstants=5, b0),"   /* For deltaTime and gravity */ \
	"DescriptorTable(UAV(u0, numDescriptors = 1))" /* UAV for particle data */
