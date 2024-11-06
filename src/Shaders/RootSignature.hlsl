//#define ROOTSIG \
//"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
//"RootConstants(num32BitConstants=3, b0)"

#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=32, b0),"   // 48 floats: 16 for view matrix, 16 for projection matrix, and 16 for any additional constants