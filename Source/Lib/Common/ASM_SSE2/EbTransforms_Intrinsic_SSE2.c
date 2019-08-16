/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbTransforms_SSE2.h"
#include "EbIntrinMacros16bit_SSE2.h"
#include <emmintrin.h>

/*****************************
* Defines
*****************************/

#define MACRO_TRANS_2MAC_NO_SAVE(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT)\
    XMM_3 = _mm_load_si128((__m128i *)(TransformAsmConst + OFFSET1));\
    XMM_4 = _mm_load_si128((__m128i *)(TransformAsmConst + OFFSET2));\
    XMM_3 = _mm_madd_epi16(XMM_3, XMM_1);\
    XMM_4 = _mm_madd_epi16(XMM_4, XMM_2);\
    XMM_3 = _mm_srai_epi32(_mm_add_epi32(XMM_4, _mm_add_epi32(XMM_3, XMM_OFFSET)), SHIFT);\
    XMM_3 = _mm_packs_epi32(XMM_3, XMM_3);

#define MACRO_TRANS_2MAC(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT, OFFSET3)\
    MACRO_TRANS_2MAC_NO_SAVE(XMM_1, XMM_2, XMM_3, XMM_4, XMM_OFFSET, OFFSET1, OFFSET2, SHIFT)\
    _mm_storel_epi64((__m128i *)(transform_coefficients+OFFSET3), XMM_3);

#define TRANS8x8_OFFSET_83_36    0
#define TRANS8x8_OFFSET_36_N83  (8 + TRANS8x8_OFFSET_83_36)
#define TRANS8x8_OFFSET_89_75   (8 + TRANS8x8_OFFSET_36_N83)
#define TRANS8x8_OFFSET_50_18   (8 + TRANS8x8_OFFSET_89_75)
#define TRANS8x8_OFFSET_75_N18  (8 + TRANS8x8_OFFSET_50_18)
#define TRANS8x8_OFFSET_N89_N50 (8 + TRANS8x8_OFFSET_75_N18)
#define TRANS8x8_OFFSET_50_N89  (8 + TRANS8x8_OFFSET_N89_N50)
#define TRANS8x8_OFFSET_18_75   (8 + TRANS8x8_OFFSET_50_N89)
#define TRANS8x8_OFFSET_18_N50  (8 + TRANS8x8_OFFSET_18_75)
#define TRANS8x8_OFFSET_75_N89  (8 + TRANS8x8_OFFSET_18_N50)
#define TRANS8x8_OFFSET_256     (8 + TRANS8x8_OFFSET_75_N89)
#define TRANS8x8_OFFSET_64_64   (8 + TRANS8x8_OFFSET_256)
#define TRANS8x8_OFFSET_N18_N50 (8 + TRANS8x8_OFFSET_64_64)
#define TRANS8x8_OFFSET_N75_N89 (8 + TRANS8x8_OFFSET_N18_N50)
#define TRANS8x8_OFFSET_N36_N83 (8 + TRANS8x8_OFFSET_N75_N89)
#define TRANS8x8_OFFSET_N83_N36 (8 + TRANS8x8_OFFSET_N36_N83)
#define TRANS8x8_OFFSET_36_83   (8 + TRANS8x8_OFFSET_N83_N36)
#define TRANS8x8_OFFSET_50_89   (8 + TRANS8x8_OFFSET_36_83)
#define TRANS8x8_OFFSET_18_N75  (8 + TRANS8x8_OFFSET_50_89)
#define TRANS8x8_OFFSET_N64_64  (8 + TRANS8x8_OFFSET_18_N75)
#define TRANS8x8_OFFSET_64_N64  (8 + TRANS8x8_OFFSET_N64_64)
#define TRANS8x8_OFFSET_N75_N18 (8 + TRANS8x8_OFFSET_64_N64)
#define TRANS8x8_OFFSET_89_N50  (8 + TRANS8x8_OFFSET_N75_N18)
#define TRANS8x8_OFFSET_83_N36  (8 + TRANS8x8_OFFSET_89_N50)
#define TRANS8x8_OFFSET_N36_83  (8 + TRANS8x8_OFFSET_83_N36)
#define TRANS8x8_OFFSET_N83_36  (8 + TRANS8x8_OFFSET_N36_83)
#define TRANS8x8_OFFSET_89_N75  (8 + TRANS8x8_OFFSET_N83_36)
#define TRANS8x8_OFFSET_50_N18  (8 + TRANS8x8_OFFSET_89_N75)

#define MACRO_CALC_EVEN_ODD(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8)\
    even0 = _mm_add_epi16(XMM1, XMM8);\
    even1 = _mm_add_epi16(XMM2, XMM7);\
    even2 = _mm_add_epi16(XMM3, XMM6);\
    even3 = _mm_add_epi16(XMM4, XMM5);\
    odd0 = _mm_sub_epi16(XMM1, XMM8);\
    odd1 = _mm_sub_epi16(XMM2, XMM7);\
    odd2 = _mm_sub_epi16(XMM3, XMM6);\
    odd3 = _mm_sub_epi16(XMM4, XMM5);

#define MACRO_TRANS_4MAC_NO_SAVE(XMM1, XMM2, XMM3, XMM4, XMM_RET, XMM_OFFSET, MEM, OFFSET1, OFFSET2, SHIFT)\
    XMM_RET = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(XMM1, _mm_load_si128((__m128i*)(MEM+OFFSET1))),\
                                                                         _mm_madd_epi16(XMM3, _mm_load_si128((__m128i*)(MEM+OFFSET2)))), XMM_OFFSET), SHIFT),\
                              _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(XMM2, _mm_load_si128((__m128i*)(MEM+OFFSET1))),\
                                                                         _mm_madd_epi16(XMM4, _mm_load_si128((__m128i*)(MEM+OFFSET2)))), XMM_OFFSET), SHIFT));

#define MACRO_TRANS_8MAC(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);\
    sum = _mm_packs_epi32(sum1, sum3);\
    INSTR((__m128i *)(DST + OFST5), sum);

#define MACRO_TRANS_8MAC_PF_N2(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    /*sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));*/\
    /*sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));*/\
    /*sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);*/\
    /*sum = _mm_packs_epi32(sum1, sum3);*/\
    sum = _mm_packs_epi32(sum1, sum1);\
    INSTR((__m128i *)(DST + OFST5), sum);
#define MACRO_TRANS_8MAC_PF_N4(XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM_OFST, MEM, OFST1, OFST2, OFST3, OFST4, SHIFT, INSTR, DST, OFST5)\
    sum1 = _mm_add_epi32(_mm_madd_epi16(XMM1, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM2, _mm_loadu_si128((__m128i *)(MEM + OFST2))));\
    sum2 = _mm_add_epi32(_mm_madd_epi16(XMM3, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM4, _mm_loadu_si128((__m128i *)(MEM + OFST4))));\
    sum1 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum1, sum2)), SHIFT);\
    /*sum3 = _mm_add_epi32(_mm_madd_epi16(XMM5, _mm_loadu_si128((__m128i *)(MEM + OFST1))), _mm_madd_epi16(XMM6, _mm_loadu_si128((__m128i *)(MEM + OFST2))));*/\
    /*sum4 = _mm_add_epi32(_mm_madd_epi16(XMM7, _mm_loadu_si128((__m128i *)(MEM + OFST3))), _mm_madd_epi16(XMM8, _mm_loadu_si128((__m128i *)(MEM + OFST4))));*/\
    /*sum3 = _mm_srai_epi32(_mm_add_epi32(XMM_OFST, _mm_add_epi32(sum3, sum4)), SHIFT);*/\
    /*sum = _mm_packs_epi32(sum1, sum3);*/\
    sum = _mm_packs_epi32(sum1, sum1);\
    INSTR((__m128i *)(DST + OFST5), sum);

#if defined(__linux__) || defined(__APPLE__)
#ifndef __cplusplus
__attribute__((visibility("hidden")))
#endif
#endif
EB_ALIGN(16) const int16_t dst_transform_asm_const_sse2[] = {
    1, 0, 1, 0, 1, 0, 1, 0,
    29, 55, 29, 55, 29, 55, 29, 55,
    74, 84, 74, 84, 74, 84, 74, 84,
    84, -29, 84, -29, 84, -29, 84, -29,
    -74, 55, -74, 55, -74, 55, -74, 55,
    55, -84, 55, -84, 55, -84, 55, -84,
    74, -29, 74, -29, 74, -29, 74, -29,
    37, 37, 37, 37, 37, 37, 37, 37,
    74, 74, 74, 74, 74, 74, 74, 74,
    0, -37, 0, -37, 0, -37, 0, -37,
    0, -74, 0, -74, 0, -74, 0, -74,
    //74,    0,   74,    0,   74,    0,   74,    0,
    //55,  -29,   55,  -29,   55,  -29,   55,  -29,
};

#if defined(__linux__) || defined(__APPLE__)
#ifndef __cplusplus
__attribute__((visibility("hidden")))
#endif
#endif
EB_ALIGN(16) const int16_t inv_transform_asm_const_sse2[] = {
    2, 0, 2, 0, 2, 0, 2, 0,
    4, 0, 4, 0, 4, 0, 4, 0,
    8, 0, 8, 0, 8, 0, 8, 0,
    9, 0, 9, 0, 9, 0, 9, 0,
    64, 0, 64, 0, 64, 0, 64, 0,
    256, 0, 256, 0, 256, 0, 256, 0,
    512, 0, 512, 0, 512, 0, 512, 0,
    1024, 0, 1024, 0, 1024, 0, 1024, 0,
    2048, 0, 2048, 0, 2048, 0, 2048, 0,
    7, 0, 0, 0, 0, 0, 0, 0,
    12, 0, 0, 0, 0, 0, 0, 0,
    64, 64, 64, 64, 64, 64, 64, 64,
    90, 57, 90, 57, 90, 57, 90, 57,
    89, 50, 89, 50, 89, 50, 89, 50,
    87, 43, 87, 43, 87, 43, 87, 43,
    83, 36, 83, 36, 83, 36, 83, 36,
    80, 25, 80, 25, 80, 25, 80, 25,
    75, 18, 75, 18, 75, 18, 75, 18,
    70, 9, 70, 9, 70, 9, 70, 9,
    64, -64, 64, -64, 64, -64, 64, -64,
    87, -80, 87, -80, 87, -80, 87, -80,
    75, -89, 75, -89, 75, -89, 75, -89,
    57, -90, 57, -90, 57, -90, 57, -90,
    36, -83, 36, -83, 36, -83, 36, -83,
    9, -70, 9, -70, 9, -70, 9, -70,
    -18, -50, -18, -50, -18, -50, -18, -50,
    -43, -25, -43, -25, -43, -25, -43, -25,
    80, -25, 80, -25, 80, -25, 80, -25,
    50, 18, 50, 18, 50, 18, 50, 18,
    9, 57, 9, 57, 9, 57, 9, 57,
    -36, 83, -36, 83, -36, 83, -36, 83,
    -70, 90, -70, 90, -70, 90, -70, 90,
    -89, 75, -89, 75, -89, 75, -89, 75,
    -87, 43, -87, 43, -87, 43, -87, 43,
    70, 90, 70, 90, 70, 90, 70, 90,
    18, 75, 18, 75, 18, 75, 18, 75,
    -43, 25, -43, 25, -43, 25, -43, 25,
    -83, -36, -83, -36, -83, -36, -83, -36,
    -87, -80, -87, -80, -87, -80, -87, -80,
    -50, -89, -50, -89, -50, -89, -50, -89,
    9, -57, 9, -57, 9, -57, 9, -57,
    57, -9, 57, -9, 57, -9, 57, -9,
    -18, -75, -18, -75, -18, -75, -18, -75,
    -80, -87, -80, -87, -80, -87, -80, -87,
    -25, 43, -25, 43, -25, 43, -25, 43,
    50, 89, 50, 89, 50, 89, 50, 89,
    90, 70, 90, 70, 90, 70, 90, 70,
    43, -87, 43, -87, 43, -87, 43, -87,
    -50, -18, -50, -18, -50, -18, -50, -18,
    -90, 70, -90, 70, -90, 70, -90, 70,
    57, 9, 57, 9, 57, 9, 57, 9,
    89, -75, 89, -75, 89, -75, 89, -75,
    25, -80, 25, -80, 25, -80, 25, -80,
    25, 43, 25, 43, 25, 43, 25, 43,
    -75, 89, -75, 89, -75, 89, -75, 89,
    -70, 9, -70, 9, -70, 9, -70, 9,
    90, -57, 90, -57, 90, -57, 90, -57,
    18, 50, 18, 50, 18, 50, 18, 50,
    -80, 87, -80, 87, -80, 87, -80, 87,
    9, 70, 9, 70, 9, 70, 9, 70,
    -89, -50, -89, -50, -89, -50, -89, -50,
    -25, -80, -25, -80, -25, -80, -25, -80,
    43, 87, 43, 87, 43, 87, 43, 87,
    -75, -18, -75, -18, -75, -18, -75, -18,
    -57, -90, -57, -90, -57, -90, -57, -90,
    -9, -70, -9, -70, -9, -70, -9, -70,
    25, 80, 25, 80, 25, 80, 25, 80,
    -43, -87, -43, -87, -43, -87, -43, -87,
    57, 90, 57, 90, 57, 90, 57, 90,
    -25, -43, -25, -43, -25, -43, -25, -43,
    70, -9, 70, -9, 70, -9, 70, -9,
    -90, 57, -90, 57, -90, 57, -90, 57,
    80, -87, 80, -87, 80, -87, 80, -87,
    -43, 87, -43, 87, -43, 87, -43, 87,
    90, -70, 90, -70, 90, -70, 90, -70,
    -57, -9, -57, -9, -57, -9, -57, -9,
    -25, 80, -25, 80, -25, 80, -25, 80,
    -57, 9, -57, 9, -57, 9, -57, 9,
    80, 87, 80, 87, 80, 87, 80, 87,
    25, -43, 25, -43, 25, -43, 25, -43,
    -90, -70, -90, -70, -90, -70, -90, -70,
    -70, -90, -70, -90, -70, -90, -70, -90,
    43, -25, 43, -25, 43, -25, 43, -25,
    87, 80, 87, 80, 87, 80, 87, 80,
    -9, 57, -9, 57, -9, 57, -9, 57,
    -80, 25, -80, 25, -80, 25, -80, 25,
    -9, -57, -9, -57, -9, -57, -9, -57,
    70, -90, 70, -90, 70, -90, 70, -90,
    87, -43, 87, -43, 87, -43, 87, -43,
    -87, 80, -87, 80, -87, 80, -87, 80,
    -57, 90, -57, 90, -57, 90, -57, 90,
    -9, 70, -9, 70, -9, 70, -9, 70,
    43, 25, 43, 25, 43, 25, 43, 25,
    -90, -57, -90, -57, -90, -57, -90, -57,
    -87, -43, -87, -43, -87, -43, -87, -43,
    -80, -25, -80, -25, -80, -25, -80, -25,
    -70, -9, -70, -9, -70, -9, -70, -9,
    90, 61, 90, 61, 90, 61, 90, 61,
    90, 54, 90, 54, 90, 54, 90, 54,
    88, 46, 88, 46, 88, 46, 88, 46,
    85, 38, 85, 38, 85, 38, 85, 38,
    82, 31, 82, 31, 82, 31, 82, 31,
    78, 22, 78, 22, 78, 22, 78, 22,
    73, 13, 73, 13, 73, 13, 73, 13,
    67, 4, 67, 4, 67, 4, 67, 4,
    90, -73, 90, -73, 90, -73, 90, -73,
    82, -85, 82, -85, 82, -85, 82, -85,
    67, -90, 67, -90, 67, -90, 67, -90,
    46, -88, 46, -88, 46, -88, 46, -88,
    22, -78, 22, -78, 22, -78, 22, -78,
    -4, -61, -4, -61, -4, -61, -4, -61,
    -31, -38, -31, -38, -31, -38, -31, -38,
    -54, -13, -54, -13, -54, -13, -54, -13,
    88, -46, 88, -46, 88, -46, 88, -46,
    67, -4, 67, -4, 67, -4, 67, -4,
    31, 38, 31, 38, 31, 38, 31, 38,
    -13, 73, -13, 73, -13, 73, -13, 73,
    -54, 90, -54, 90, -54, 90, -54, 90,
    -82, 85, -82, 85, -82, 85, -82, 85,
    -90, 61, -90, 61, -90, 61, -90, 61,
    -78, 22, -78, 22, -78, 22, -78, 22,
    85, 82, 85, 82, 85, 82, 85, 82,
    46, 88, 46, 88, 46, 88, 46, 88,
    -13, 54, -13, 54, -13, 54, -13, 54,
    -67, -4, -67, -4, -67, -4, -67, -4,
    -90, -61, -90, -61, -90, -61, -90, -61,
    -73, -90, -73, -90, -73, -90, -73, -90,
    -22, -78, -22, -78, -22, -78, -22, -78,
    38, -31, 38, -31, 38, -31, 38, -31,
    22, -46, 22, -46, 22, -46, 22, -46,
    -54, -90, -54, -90, -54, -90, -54, -90,
    -90, -67, -90, -67, -90, -67, -90, -67,
    -61, 4, -61, 4, -61, 4, -61, 4,
    13, 73, 13, 73, 13, 73, 13, 73,
    78, 88, 78, 88, 78, 88, 78, 88,
    78, -88, 78, -88, 78, -88, 78, -88,
    -82, 31, -82, 31, -82, 31, -82, 31,
    -73, 90, -73, 90, -73, 90, -73, 90,
    13, 54, 13, 54, 13, 54, 13, 54,
    85, -38, 85, -38, 85, -38, 85, -38,
    -22, -46, -22, -46, -22, -46, -22, -46,
    73, -13, 73, -13, 73, -13, 73, -13,
    -31, 82, -31, 82, -31, 82, -31, 82,
    -38, 85, -38, 85, -38, 85, -38, 85,
    -90, 54, -90, 54, -90, 54, -90, 54,
    67, 90, 67, 90, 67, 90, 67, 90,
    -54, 13, -54, 13, -54, 13, -54, 13,
    -78, -88, -78, -88, -78, -88, -78, -88,
    -22, 46, -22, 46, -22, 46, -22, 46,
    -90, -73, -90, -73, -90, -73, -90, -73,
    4, -61, 4, -61, 4, -61, 4, -61,
    61, -4, 61, -4, 61, -4, 61, -4,
    -46, 22, -46, 22, -46, 22, -46, 22,
    82, 85, 82, 85, 82, 85, 82, 85,
    31, -38, 31, -38, 31, -38, 31, -38,
    -88, -78, -88, -78, -88, -78, -88, -78,
    90, 67, 90, 67, 90, 67, 90, 67,
    54, -90, 54, -90, 54, -90, 54, -90,
    -85, 38, -85, 38, -85, 38, -85, 38,
    -4, 67, -4, 67, -4, 67, -4, 67,
    88, -78, 88, -78, 88, -78, 88, -78,
    -46, -22, -46, -22, -46, -22, -46, -22,
    -61, 90, -61, 90, -61, 90, -61, 90,
    82, -31, 82, -31, 82, -31, 82, -31,
    13, -73, 13, -73, 13, -73, 13, -73,
    46, 22, 46, 22, 46, 22, 46, 22,
    -90, 67, -90, 67, -90, 67, -90, 67,
    38, -85, 38, -85, 38, -85, 38, -85,
    54, 13, 54, 13, 54, 13, 54, 13,
    -90, 73, -90, 73, -90, 73, -90, 73,
    31, -82, 31, -82, 31, -82, 31, -82,
    61, 4, 61, 4, 61, 4, 61, 4,
    -88, 78, -88, 78, -88, 78, -88, 78,
    38, 85, 38, 85, 38, 85, 38, 85,
    -4, 61, -4, 61, -4, 61, -4, 61,
    -67, -90, -67, -90, -67, -90, -67, -90,
    -31, -82, -31, -82, -31, -82, -31, -82,
    -78, -22, -78, -22, -78, -22, -78, -22,
    90, 73, 90, 73, 90, 73, 90, 73,
    -61, -90, -61, -90, -61, -90, -61, -90,
    4, 67, 4, 67, 4, 67, 4, 67,
    54, -13, 54, -13, 54, -13, 54, -13,
    -88, -46, -88, -46, -88, -46, -88, -46,
    85, -82, 85, -82, 85, -82, 85, -82,
    -38, -31, -38, -31, -38, -31, -38, -31,
    -13, -73, -13, -73, -13, -73, -13, -73,
    22, 78, 22, 78, 22, 78, 22, 78,
    -46, -88, -46, -88, -46, -88, -46, -88,
    54, 90, 54, 90, 54, 90, 54, 90
};

#if defined(__linux__) || defined(__APPLE__)
#ifndef __cplusplus
__attribute__((visibility("hidden")))
#endif
#endif
EB_ALIGN(16) const int16_t inv_dst_transform_asm_const_sse2[] = {
    64, 0, 64, 0, 64, 0, 64, 0,
    29, 84, 29, 84, 29, 84, 29, 84,
    74, 55, 74, 55, 74, 55, 74, 55,
    55, -29, 55, -29, 55, -29, 55, -29,
    74, -84, 74, -84, 74, -84, 74, -84,
    74, -74, 74, -74, 74, -74, 74, -74,
    0, 74, 0, 74, 0, 74, 0, 74,
    84, 55, 84, 55, 84, 55, 84, 55,
    -74, -29, -74, -29, -74, -29, -74, -29,
};

// Coefficients for inverse 32-point transform
EB_EXTERN const int16_t coeff_tbl2[48 * 8] =
{
    64, 89, 64, 75, 64, 50, 64, 18, 64, -18, 64, -50, 64, -75, 64, -89,
    83, 75, 36, -18, -36, -89, -83, -50, -83, 50, -36, 89, 36, 18, 83, -75,
    64, 50, -64, -89, -64, 18, 64, 75, 64, -75, -64, -18, -64, 89, 64, -50,
    36, 18, -83, -50, 83, 75, -36, -89, -36, 89, 83, -75, -83, 50, 36, -18,
    90, 87, 87, 57, 80, 9, 70, -43, 57, -80, 43, -90, 25, -70, 9, -25,
    80, 70, 9, -43, -70, -87, -87, 9, -25, 90, 57, 25, 90, -80, 43, -57,
    57, 43, -80, -90, -25, 57, 90, 25, -9, -87, -87, 70, 43, 9, 70, -80,
    25, 9, -70, -25, 90, 43, -80, -57, 43, 70, 9, -80, -57, 87, 87, -90,
    90, 90, 90, 82, 88, 67, 85, 46, 82, 22, 78, -4, 73, -31, 67, -54,
    61, -73, 54, -85, 46, -90, 38, -88, 31, -78, 22, -61, 13, -38, 4, -13,
    88, 85, 67, 46, 31, -13, -13, -67, -54, -90, -82, -73, -90, -22, -78, 38,
    -46, 82, -4, 88, 38, 54, 73, -4, 90, -61, 85, -90, 61, -78, 22, -31,
    82, 78, 22, -4, -54, -82, -90, -73, -61, 13, 13, 85, 78, 67, 85, -22,
    31, -88, -46, -61, -90, 31, -67, 90, 4, 54, 73, -38, 88, -90, 38, -46,
    73, 67, -31, -54, -90, -78, -22, 38, 78, 85, 67, -22, -38, -90, -90, 4,
    -13, 90, 82, 13, 61, -88, -46, -31, -88, 82, -4, 46, 85, -73, 54, -61,
    61, 54, -73, -85, -46, -4, 82, 88, 31, -46, -88, -61, -13, 82, 90, 13,
    -4, -90, -90, 38, 22, 67, 85, -78, -38, -22, -78, 90, 54, -31, 67, -73,
    46, 38, -90, -88, 38, 73, 54, -4, -90, -67, 31, 90, 61, -46, -88, -31,
    22, 85, 67, -78, -85, 13, 13, 61, 73, -90, -82, 54, 4, 22, 78, -82,
    31, 22, -78, -61, 90, 85, -61, -90, 4, 73, 54, -38, -88, -4, 82, 46,
    -38, -78, -22, 90, 73, -82, -90, 54, 67, -13, -13, -31, -46, 67, 85, -88,
    13, 4, -38, -13, 61, 22, -78, -31, 88, 38, -90, -46, 85, 54, -73, -61,
    54, 67, -31, -73, 4, 78, 22, -82, -46, 85, 67, -88, -82, 90, 90, -90
};

#if defined(__linux__) || defined(__APPLE__)
#ifndef __cplusplus
__attribute__((visibility("hidden")))
#endif
#endif
EB_EXTERN const int16_t coeff_tbl[48 * 8] =
{
    64, 64, 89, 75, 83, 36, 75, -18, 64, -64, 50, -89, 36, -83, 18, -50,
    64, 64, 50, 18, -36, -83, -89, -50, -64, 64, 18, 75, 83, -36, 75, -89,
    64, 64, -18, -50, -83, -36, 50, 89, 64, -64, -75, -18, -36, 83, 89, -75,
    64, 64, -75, -89, 36, 83, 18, -75, -64, 64, 89, -50, -83, 36, 50, -18,
    90, 87, 87, 57, 80, 9, 70, -43, 57, -80, 43, -90, 25, -70, 9, -25,
    80, 70, 9, -43, -70, -87, -87, 9, -25, 90, 57, 25, 90, -80, 43, -57,
    57, 43, -80, -90, -25, 57, 90, 25, -9, -87, -87, 70, 43, 9, 70, -80,
    25, 9, -70, -25, 90, 43, -80, -57, 43, 70, 9, -80, -57, 87, 87, -90,
    90, 90, 90, 82, 88, 67, 85, 46, 82, 22, 78, -4, 73, -31, 67, -54,
    61, -73, 54, -85, 46, -90, 38, -88, 31, -78, 22, -61, 13, -38, 4, -13,
    88, 85, 67, 46, 31, -13, -13, -67, -54, -90, -82, -73, -90, -22, -78, 38,
    -46, 82, -4, 88, 38, 54, 73, -4, 90, -61, 85, -90, 61, -78, 22, -31,
    82, 78, 22, -4, -54, -82, -90, -73, -61, 13, 13, 85, 78, 67, 85, -22,
    31, -88, -46, -61, -90, 31, -67, 90, 4, 54, 73, -38, 88, -90, 38, -46,
    73, 67, -31, -54, -90, -78, -22, 38, 78, 85, 67, -22, -38, -90, -90, 4,
    -13, 90, 82, 13, 61, -88, -46, -31, -88, 82, -4, 46, 85, -73, 54, -61,
    61, 54, -73, -85, -46, -4, 82, 88, 31, -46, -88, -61, -13, 82, 90, 13,
    -4, -90, -90, 38, 22, 67, 85, -78, -38, -22, -78, 90, 54, -31, 67, -73,
    46, 38, -90, -88, 38, 73, 54, -4, -90, -67, 31, 90, 61, -46, -88, -31,
    22, 85, 67, -78, -85, 13, 13, 61, 73, -90, -82, 54, 4, 22, 78, -82,
    31, 22, -78, -61, 90, 85, -61, -90, 4, 73, 54, -38, -88, -4, 82, 46,
    -38, -78, -22, 90, 73, -82, -90, 54, 67, -13, -13, -31, -46, 67, 85, -88,
    13, 4, -38, -13, 61, 22, -78, -31, 88, 38, -90, -46, 85, 54, -73, -61,
    54, 67, -31, -73, 4, 78, 22, -82, -46, 85, 67, -88, -82, 90, 90, -90
};

static __m128i reverse_epi16(__m128i x)
{
    x = _mm_shuffle_epi32(x, 0x1b); // 00011011
    x = _mm_shufflelo_epi16(x, 0xb1); // 10110001
    x = _mm_shufflehi_epi16(x, 0xb1);
    return x;
}

// 16-point forward transform (16 rows)
static void Transform16(short *src, int32_t src_stride, short *dst, int32_t dst_stride, int32_t shift)
{
    int32_t i;
    __m128i s0 = _mm_cvtsi32_si128(shift);
    __m128i o0 = _mm_set1_epi32(1 << (shift - 1));
    const __m128i *coeff32 = (const __m128i *)coeff_tbl;

    for (i = 0; i < 16; i++)
    {
        __m128i x0, x1;
        __m128i y0, y1;
        __m128i a0, a1, a2, a3;
        __m128i b0, b1, b2, b3;

        y0 = _mm_loadu_si128((const __m128i *)(src + i * src_stride + 0x00));
        y1 = _mm_loadu_si128((const __m128i *)(src + i * src_stride + 0x08));

        // 16-point butterfly
        y1 = reverse_epi16(y1);

        x0 = _mm_add_epi16(y0, y1);
        x1 = _mm_sub_epi16(y0, y1);

        a0 = _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xff), coeff32[6]));

        a1 = _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x00), coeff32[1]);
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x55), coeff32[3]));
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xaa), coeff32[5]));
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xff), coeff32[14]));

        a3 = _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x00), coeff32[9]);
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x55), coeff32[11]));
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xaa), coeff32[13]));
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xff), coeff32[15]));

        b0 = _mm_sra_epi32(_mm_add_epi32(a0, o0), s0);
        b1 = _mm_sra_epi32(_mm_add_epi32(a1, o0), s0);
        b2 = _mm_sra_epi32(_mm_add_epi32(a2, o0), s0);
        b3 = _mm_sra_epi32(_mm_add_epi32(a3, o0), s0);

        x0 = _mm_packs_epi32(b0, b1);
        x1 = _mm_packs_epi32(b2, b3);

        y0 = _mm_unpacklo_epi16(x0, x1);
        y1 = _mm_unpackhi_epi16(x0, x1);

        _mm_storeu_si128((__m128i *)(dst + i * dst_stride + 0x00), y0);
        _mm_storeu_si128((__m128i *)(dst + i * dst_stride + 0x08), y1);
    }
}

// 16-point inverse transform (16 rows)
static void InvTransform16(
    int16_t  *src,
    uint32_t   src_stride,
    int16_t  *dst,
    uint32_t   dst_stride,
    uint32_t   shift)
{
    int32_t i;
    __m128i s0 = _mm_cvtsi32_si128(shift);
    __m128i o0 = _mm_set1_epi32(1 << (shift - 1));
    const __m128i *coeff32 = (const __m128i *)coeff_tbl2;

    for (i = 0; i < 16; i++)
    {
        __m128i x0, x1;
        __m128i y0, y1;
        __m128i a0, a1, a2, a3;
        __m128i b0, b1, b2, b3;
        x0 = _mm_loadu_si128((const __m128i *)(src + i * src_stride + 0x00)); // 00 01 02 03 04 05 06 07
        x1 = _mm_loadu_si128((const __m128i *)(src + i * src_stride + 0x08)); // 08 09 0a 0b 0c 0d 0e 0f

        y0 = _mm_unpacklo_epi16(x0, x1); // 00 08 01 09 02 0a 03 0b
        y1 = _mm_unpackhi_epi16(x0, x1); // 04 0c 05 0d 06 0e 07 0f

        x0 = _mm_unpacklo_epi16(y0, y1); // 00 04 08 0c 01 05 09 0d
        x1 = _mm_unpackhi_epi16(y0, y1); // 02 06 0a 0e 03 07 0b 0f

        y0 = _mm_unpacklo_epi16(x0, x1); // 00 02 04 06 08 0a 0c 0e
        y1 = _mm_unpackhi_epi16(x0, x1); // 01 03 05 07 09 0b 0d 0f

        x0 = y0;
        x1 = y1;

        a0 = _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x00), coeff32[0]);
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x55), coeff32[2]));
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xaa), coeff32[4]));
        a0 = _mm_add_epi32(a0, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xff), coeff32[6]));

        a1 = _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x00), coeff32[1]);
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0x55), coeff32[3]));
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xaa), coeff32[5]));
        a1 = _mm_add_epi32(a1, _mm_madd_epi16(_mm_shuffle_epi32(x0, 0xff), coeff32[7]));

        a2 = _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x00), coeff32[8]);
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x55), coeff32[10]));
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xaa), coeff32[12]));
        a2 = _mm_add_epi32(a2, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xff), coeff32[14]));

        a3 = _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x00), coeff32[9]);
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0x55), coeff32[11]));
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xaa), coeff32[13]));
        a3 = _mm_add_epi32(a3, _mm_madd_epi16(_mm_shuffle_epi32(x1, 0xff), coeff32[15]));

        a0 = _mm_add_epi32(a0, o0);
        a1 = _mm_add_epi32(a1, o0);

        b0 = _mm_add_epi32(a0, a2);
        b1 = _mm_add_epi32(a1, a3);
        b2 = _mm_sub_epi32(a0, a2);
        b3 = _mm_sub_epi32(a1, a3);

        a0 = b0;
        a1 = b1;
        a2 = _mm_shuffle_epi32(b3, 0x1b); // 00011011
        a3 = _mm_shuffle_epi32(b2, 0x1b);

        a0 = _mm_sra_epi32(a0, s0);
        a1 = _mm_sra_epi32(a1, s0);
        a2 = _mm_sra_epi32(a2, s0);
        a3 = _mm_sra_epi32(a3, s0);

        x0 = _mm_packs_epi32(a0, a1);
        x1 = _mm_packs_epi32(a2, a3);

        _mm_storeu_si128((__m128i *)(dst + i * dst_stride + 0x00), x0);
        _mm_storeu_si128((__m128i *)(dst + i * dst_stride + 0x08), x1);
    }
}

// transpose 16x16 block of data
static void Transpose16(
    int16_t     *src,
    uint32_t      src_stride,
    int16_t     *dst,
    uint32_t      dst_stride)
{
    int32_t i, j;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            __m128i a0, a1, a2, a3, a4, a5, a6, a7;
            __m128i b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 0)*src_stride + 8 * j));
            a1 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 1)*src_stride + 8 * j));
            a2 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 2)*src_stride + 8 * j));
            a3 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 3)*src_stride + 8 * j));
            a4 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 4)*src_stride + 8 * j));
            a5 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 5)*src_stride + 8 * j));
            a6 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 6)*src_stride + 8 * j));
            a7 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 7)*src_stride + 8 * j));

            b0 = _mm_unpacklo_epi16(a0, a4);
            b1 = _mm_unpacklo_epi16(a1, a5);
            b2 = _mm_unpacklo_epi16(a2, a6);
            b3 = _mm_unpacklo_epi16(a3, a7);
            b4 = _mm_unpackhi_epi16(a0, a4);
            b5 = _mm_unpackhi_epi16(a1, a5);
            b6 = _mm_unpackhi_epi16(a2, a6);
            b7 = _mm_unpackhi_epi16(a3, a7);

            a0 = _mm_unpacklo_epi16(b0, b2);
            a1 = _mm_unpacklo_epi16(b1, b3);
            a2 = _mm_unpackhi_epi16(b0, b2);
            a3 = _mm_unpackhi_epi16(b1, b3);
            a4 = _mm_unpacklo_epi16(b4, b6);
            a5 = _mm_unpacklo_epi16(b5, b7);
            a6 = _mm_unpackhi_epi16(b4, b6);
            a7 = _mm_unpackhi_epi16(b5, b7);

            b0 = _mm_unpacklo_epi16(a0, a1);
            b1 = _mm_unpackhi_epi16(a0, a1);
            b2 = _mm_unpacklo_epi16(a2, a3);
            b3 = _mm_unpackhi_epi16(a2, a3);
            b4 = _mm_unpacklo_epi16(a4, a5);
            b5 = _mm_unpackhi_epi16(a4, a5);
            b6 = _mm_unpacklo_epi16(a6, a7);
            b7 = _mm_unpackhi_epi16(a6, a7);

            _mm_storeu_si128((__m128i *)(dst + (8 * j + 0)*dst_stride + 8 * i), b0);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 1)*dst_stride + 8 * i), b1);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 2)*dst_stride + 8 * i), b2);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 3)*dst_stride + 8 * i), b3);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 4)*dst_stride + 8 * i), b4);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 5)*dst_stride + 8 * i), b5);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 6)*dst_stride + 8 * i), b6);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 7)*dst_stride + 8 * i), b7);
        }
    }
}

void PfreqTranspose32Type1_SSE2(
    int16_t *src,
    uint32_t  src_stride,
    int16_t *dst,
    uint32_t  dst_stride)
{
    uint32_t i, j;
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            __m128i a0, a1, a2, a3, a4, a5, a6, a7;
            __m128i b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 0)*src_stride + 8 * j));
            a1 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 1)*src_stride + 8 * j));
            a2 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 2)*src_stride + 8 * j));
            a3 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 3)*src_stride + 8 * j));
            a4 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 4)*src_stride + 8 * j));
            a5 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 5)*src_stride + 8 * j));
            a6 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 6)*src_stride + 8 * j));
            a7 = _mm_loadu_si128((const __m128i *)(src + (8 * i + 7)*src_stride + 8 * j));

            b0 = _mm_unpacklo_epi16(a0, a4);
            b1 = _mm_unpacklo_epi16(a1, a5);
            b2 = _mm_unpacklo_epi16(a2, a6);
            b3 = _mm_unpacklo_epi16(a3, a7);
            b4 = _mm_unpackhi_epi16(a0, a4);
            b5 = _mm_unpackhi_epi16(a1, a5);
            b6 = _mm_unpackhi_epi16(a2, a6);
            b7 = _mm_unpackhi_epi16(a3, a7);

            a0 = _mm_unpacklo_epi16(b0, b2);
            a1 = _mm_unpacklo_epi16(b1, b3);
            a2 = _mm_unpackhi_epi16(b0, b2);
            a3 = _mm_unpackhi_epi16(b1, b3);
            a4 = _mm_unpacklo_epi16(b4, b6);
            a5 = _mm_unpacklo_epi16(b5, b7);
            a6 = _mm_unpackhi_epi16(b4, b6);
            a7 = _mm_unpackhi_epi16(b5, b7);

            b0 = _mm_unpacklo_epi16(a0, a1);
            b1 = _mm_unpackhi_epi16(a0, a1);
            b2 = _mm_unpacklo_epi16(a2, a3);
            b3 = _mm_unpackhi_epi16(a2, a3);
            b4 = _mm_unpacklo_epi16(a4, a5);
            b5 = _mm_unpackhi_epi16(a4, a5);
            b6 = _mm_unpacklo_epi16(a6, a7);
            b7 = _mm_unpackhi_epi16(a6, a7);

            _mm_storeu_si128((__m128i *)(dst + (8 * j + 0)*dst_stride + 8 * i), b0);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 1)*dst_stride + 8 * i), b1);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 2)*dst_stride + 8 * i), b2);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 3)*dst_stride + 8 * i), b3);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 4)*dst_stride + 8 * i), b4);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 5)*dst_stride + 8 * i), b5);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 6)*dst_stride + 8 * i), b6);
            _mm_storeu_si128((__m128i *)(dst + (8 * j + 7)*dst_stride + 8 * i), b7);
        }
    }
}
