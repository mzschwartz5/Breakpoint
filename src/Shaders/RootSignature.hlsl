#define ROOTSIG \
"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," \
"RootConstants(num32BitConstants=32, b0),"   /* For matrices */ \
"DescriptorTable(CBV(b1))"                   /* TODO: Change to RootCBV instead of Descriptor Table for model matrices buffer */