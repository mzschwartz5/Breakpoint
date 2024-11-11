#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=32, b0),"   /* For view and projection matrices */ \
"DescriptorTable(CBV(b1))"                   /* CBV for model matrices buffer */