//#define ROOTSIG \
//"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
//"RootConstants(num32BitConstants=3, b0)"

#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"DescriptorTable(UAV(u0))"                   /* SRV for model matrices buffer */