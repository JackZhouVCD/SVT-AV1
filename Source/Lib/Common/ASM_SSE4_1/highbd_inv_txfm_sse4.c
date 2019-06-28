/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include <assert.h>
#include <smmintrin.h> /* SSE4.1 */

#include "EbDefinitions.h"
#include "aom_dsp_rtcd.h"

#include "highbd_txfm_utility_sse4.h"
#include "EbTransforms.h"

const int32_t *cospi_arr(int32_t n);
const int32_t *sinpi_arr(int32_t n);
extern const int8_t *inv_txfm_shift_ls[];

#if AVX2_ADD
#else
#define NewSqrt2Bits ((int32_t)12)
// 2^12 * sqrt(2)
static const int32_t NewSqrt2 = 5793;
// 2^12 / sqrt(2)
static const int32_t NewInvSqrt2 = 2896;
#endif

typedef enum ATTRIBUTE_PACKED {
    IDCT_1D,
    IADST_1D,
    IFLIPADST_1D = IADST_1D,
    IIDENTITY_1D,
    ITX_TYPES_1D,
} ITX_TYPE_1D;

static const ITX_TYPE_1D vitx_1d_tab[TX_TYPES] = {
    IDCT_1D, IADST_1D, IDCT_1D, IADST_1D,
    IFLIPADST_1D, IDCT_1D, IFLIPADST_1D, IADST_1D,
    IFLIPADST_1D, IIDENTITY_1D, IDCT_1D, IIDENTITY_1D,
    IADST_1D, IIDENTITY_1D, IFLIPADST_1D, IIDENTITY_1D,
};

static const ITX_TYPE_1D hitx_1d_tab[TX_TYPES] = {
    IDCT_1D, IDCT_1D, IADST_1D, IADST_1D,
    IDCT_1D, IFLIPADST_1D, IFLIPADST_1D, IFLIPADST_1D,
    IADST_1D, IIDENTITY_1D, IIDENTITY_1D, IDCT_1D,
    IIDENTITY_1D, IADST_1D, IIDENTITY_1D, IFLIPADST_1D,
};

//const int8_t **inv_cos_bit_row;
//const int8_t **inv_cos_bit_col;
// Transform block width in log2
//static const int32_t tx_size_wide_log2[TX_SIZES_ALL] = {
//    2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6,
//};
//
//// Transform block height in log2
//static const int32_t tx_size_high_log2[TX_SIZES_ALL] = {
//    2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4,
//};

typedef void(*transform_1d_sse4_1)(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift);

void get_flip_cfg(TxType tx_type, int32_t *ud_flip, int32_t *lr_flip);

static INLINE void load_buffer_4x4(const int32_t *coeff, __m128i *in) {
    in[0] = _mm_load_si128((const __m128i *)(coeff + 0));
    in[1] = _mm_load_si128((const __m128i *)(coeff + 4));
    in[2] = _mm_load_si128((const __m128i *)(coeff + 8));
    in[3] = _mm_load_si128((const __m128i *)(coeff + 12));
}

static void addsub_sse4_1(const __m128i in0, const __m128i in1, __m128i *out0,
    __m128i *out1, const __m128i *clamp_lo,
    const __m128i *clamp_hi) {
    __m128i a0 = _mm_add_epi32(in0, in1);
    __m128i a1 = _mm_sub_epi32(in0, in1);

    a0 = _mm_max_epi32(a0, *clamp_lo);
    a0 = _mm_min_epi32(a0, *clamp_hi);
    a1 = _mm_max_epi32(a1, *clamp_lo);
    a1 = _mm_min_epi32(a1, *clamp_hi);

    *out0 = a0;
    *out1 = a1;
}

static void addsub_no_clamp_sse4_1(const __m128i in0, const __m128i in1,
    __m128i *out0, __m128i *out1) {
    __m128i a0 = _mm_add_epi32(in0, in1);
    __m128i a1 = _mm_sub_epi32(in0, in1);

    *out0 = a0;
    *out1 = a1;
}

static void idct4x4_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
        int32_t bd, int32_t out_shift) {
    (void)*out;
    (void)do_cols;
    (void)bd;
    (void)out_shift;
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    __m128i u0, u1, u2, u3;
    __m128i v0, v1, v2, v3, x, y;

    v0 = _mm_unpacklo_epi32(in[0], in[1]);
    v1 = _mm_unpackhi_epi32(in[0], in[1]);
    v2 = _mm_unpacklo_epi32(in[2], in[3]);
    v3 = _mm_unpackhi_epi32(in[2], in[3]);

    u0 = _mm_unpacklo_epi64(v0, v2);
    u1 = _mm_unpackhi_epi64(v0, v2);
    u2 = _mm_unpacklo_epi64(v1, v3);
    u3 = _mm_unpackhi_epi64(v1, v3);

    x = _mm_mullo_epi32(u0, cospi32);
    y = _mm_mullo_epi32(u2, cospi32);
    v0 = _mm_add_epi32(x, y);
    v0 = _mm_add_epi32(v0, rnding);
    v0 = _mm_srai_epi32(v0, bit);

    v1 = _mm_sub_epi32(x, y);
    v1 = _mm_add_epi32(v1, rnding);
    v1 = _mm_srai_epi32(v1, bit);

    x = _mm_mullo_epi32(u1, cospi48);
    y = _mm_mullo_epi32(u3, cospim16);
    v2 = _mm_add_epi32(x, y);
    v2 = _mm_add_epi32(v2, rnding);
    v2 = _mm_srai_epi32(v2, bit);

    x = _mm_mullo_epi32(u1, cospi16);
    y = _mm_mullo_epi32(u3, cospi48);
    v3 = _mm_add_epi32(x, y);
    v3 = _mm_add_epi32(v3, rnding);
    v3 = _mm_srai_epi32(v3, bit);

    in[0] = _mm_add_epi32(v0, v3);
    in[1] = _mm_add_epi32(v1, v2);
    in[2] = _mm_sub_epi32(v1, v2);
    in[3] = _mm_sub_epi32(v0, v3);
}

static void iadst4x4_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
        int32_t bd, int32_t out_shift) {
    (void)*out;
    (void)do_cols;
    (void)bd;
    (void)out_shift;
    const int32_t *sinpi = sinpi_arr(bit);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const __m128i sinpi1 = _mm_set1_epi32((int32_t)sinpi[1]);
    const __m128i sinpi2 = _mm_set1_epi32((int32_t)sinpi[2]);
    const __m128i sinpi3 = _mm_set1_epi32((int32_t)sinpi[3]);
    const __m128i sinpi4 = _mm_set1_epi32((int32_t)sinpi[4]);
    __m128i t;
    __m128i s0, s1, s2, s3, s4, s5, s6, s7;
    __m128i x0, x1, x2, x3;
    __m128i u0, u1, u2, u3;
    __m128i v0, v1, v2, v3;

    v0 = _mm_unpacklo_epi32(in[0], in[1]);
    v1 = _mm_unpackhi_epi32(in[0], in[1]);
    v2 = _mm_unpacklo_epi32(in[2], in[3]);
    v3 = _mm_unpackhi_epi32(in[2], in[3]);

    x0 = _mm_unpacklo_epi64(v0, v2);
    x1 = _mm_unpackhi_epi64(v0, v2);
    x2 = _mm_unpacklo_epi64(v1, v3);
    x3 = _mm_unpackhi_epi64(v1, v3);

    s0 = _mm_mullo_epi32(x0, sinpi1);
    s1 = _mm_mullo_epi32(x0, sinpi2);
    s2 = _mm_mullo_epi32(x1, sinpi3);
    s3 = _mm_mullo_epi32(x2, sinpi4);
    s4 = _mm_mullo_epi32(x2, sinpi1);
    s5 = _mm_mullo_epi32(x3, sinpi2);
    s6 = _mm_mullo_epi32(x3, sinpi4);
    t = _mm_sub_epi32(x0, x2);
    s7 = _mm_add_epi32(t, x3);

    t = _mm_add_epi32(s0, s3);
    s0 = _mm_add_epi32(t, s5);
    t = _mm_sub_epi32(s1, s4);
    s1 = _mm_sub_epi32(t, s6);
    s3 = s2;
    s2 = _mm_mullo_epi32(s7, sinpi3);

    u0 = _mm_add_epi32(s0, s3);
    u1 = _mm_add_epi32(s1, s3);
    u2 = s2;
    t = _mm_add_epi32(s0, s1);
    u3 = _mm_sub_epi32(t, s3);

    u0 = _mm_add_epi32(u0, rnding);
    u0 = _mm_srai_epi32(u0, bit);

    u1 = _mm_add_epi32(u1, rnding);
    u1 = _mm_srai_epi32(u1, bit);

    u2 = _mm_add_epi32(u2, rnding);
    u2 = _mm_srai_epi32(u2, bit);

    u3 = _mm_add_epi32(u3, rnding);
    u3 = _mm_srai_epi32(u3, bit);

    in[0] = u0;
    in[1] = u1;
    in[2] = u2;
    in[3] = u3;
}

static INLINE void round_shift_4x4(__m128i *in, int32_t shift) {
    __m128i rnding = _mm_set1_epi32(1 << (shift - 1));

    in[0] = _mm_add_epi32(in[0], rnding);
    in[1] = _mm_add_epi32(in[1], rnding);
    in[2] = _mm_add_epi32(in[2], rnding);
    in[3] = _mm_add_epi32(in[3], rnding);

    in[0] = _mm_srai_epi32(in[0], shift);
    in[1] = _mm_srai_epi32(in[1], shift);
    in[2] = _mm_srai_epi32(in[2], shift);
    in[3] = _mm_srai_epi32(in[3], shift);
}

static INLINE __m128i highbd_clamp_epi16(__m128i u, int32_t bd) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i one = _mm_set1_epi16(1);
    const __m128i max = _mm_sub_epi16(_mm_slli_epi16(one, bd), one);
    __m128i clamped, mask;

    mask = _mm_cmpgt_epi16(u, max);
    clamped = _mm_andnot_si128(mask, u);
    mask = _mm_and_si128(mask, max);
    clamped = _mm_or_si128(mask, clamped);
    mask = _mm_cmpgt_epi16(clamped, zero);
    clamped = _mm_and_si128(clamped, mask);

    return clamped;
}

static void write_buffer_4x4(__m128i *in, uint16_t *output, int32_t stride,
    int32_t fliplr, int32_t flipud, int32_t shift, int32_t bd) {
    const __m128i zero = _mm_setzero_si128();
    __m128i u0, u1, u2, u3;
    __m128i v0, v1, v2, v3;

    round_shift_4x4(in, shift);

    v0 = _mm_loadl_epi64((__m128i const *)(output + 0 * stride));
    v1 = _mm_loadl_epi64((__m128i const *)(output + 1 * stride));
    v2 = _mm_loadl_epi64((__m128i const *)(output + 2 * stride));
    v3 = _mm_loadl_epi64((__m128i const *)(output + 3 * stride));

    v0 = _mm_unpacklo_epi16(v0, zero);
    v1 = _mm_unpacklo_epi16(v1, zero);
    v2 = _mm_unpacklo_epi16(v2, zero);
    v3 = _mm_unpacklo_epi16(v3, zero);

    if (fliplr) {
        in[0] = _mm_shuffle_epi32(in[0], 0x1B);
        in[1] = _mm_shuffle_epi32(in[1], 0x1B);
        in[2] = _mm_shuffle_epi32(in[2], 0x1B);
        in[3] = _mm_shuffle_epi32(in[3], 0x1B);
    }

    if (flipud) {
        u0 = _mm_add_epi32(in[3], v0);
        u1 = _mm_add_epi32(in[2], v1);
        u2 = _mm_add_epi32(in[1], v2);
        u3 = _mm_add_epi32(in[0], v3);
    }
    else {
        u0 = _mm_add_epi32(in[0], v0);
        u1 = _mm_add_epi32(in[1], v1);
        u2 = _mm_add_epi32(in[2], v2);
        u3 = _mm_add_epi32(in[3], v3);
    }

    v0 = _mm_packus_epi32(u0, u1);
    v2 = _mm_packus_epi32(u2, u3);

    u0 = highbd_clamp_epi16(v0, bd);
    u2 = highbd_clamp_epi16(v2, bd);

    v0 = _mm_unpacklo_epi64(u0, u0);
    v1 = _mm_unpackhi_epi64(u0, u0);
    v2 = _mm_unpacklo_epi64(u2, u2);
    v3 = _mm_unpackhi_epi64(u2, u2);

    _mm_storel_epi64((__m128i *)(output + 0 * stride), v0);
    _mm_storel_epi64((__m128i *)(output + 1 * stride), v1);
    _mm_storel_epi64((__m128i *)(output + 2 * stride), v2);
    _mm_storel_epi64((__m128i *)(output + 3 * stride), v3);
}

void av1_inv_txfm2d_add_4x4_sse4_1(const int32_t *coeff, uint16_t *output,
    int32_t stride, TxType tx_type, int32_t bd) {
    __m128i in[4];
    const int8_t *shift = inv_txfm_shift_ls[TX_4X4];
    const int32_t txw_idx = get_txw_idx(TX_4X4);
    const int32_t txh_idx = get_txh_idx(TX_4X4);

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_4x4(coeff, in);
        idct4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        idct4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_DCT:
        load_buffer_4x4(coeff, in);
        idct4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case DCT_ADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        idct4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_ADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case FLIPADST_DCT:
        load_buffer_4x4(coeff, in);
        idct4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 1, -shift[1], bd);
        break;
    case DCT_FLIPADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        idct4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 1, 1, -shift[1], bd);
        break;
    case ADST_FLIPADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case FLIPADST_ADST:
        load_buffer_4x4(coeff, in);
        iadst4x4_sse4_1(in, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, 0);
        iadst4x4_sse4_1(in, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
        write_buffer_4x4(in, output, stride, 0, 1, -shift[1], bd);
        break;
    default: assert(0);
    }
}

// 8x8
static void load_buffer_8x8(const int32_t *coeff, __m128i *in) {
    in[0] = _mm_load_si128((const __m128i *)(coeff + 0));
    in[1] = _mm_load_si128((const __m128i *)(coeff + 4));
    in[2] = _mm_load_si128((const __m128i *)(coeff + 8));
    in[3] = _mm_load_si128((const __m128i *)(coeff + 12));
    in[4] = _mm_load_si128((const __m128i *)(coeff + 16));
    in[5] = _mm_load_si128((const __m128i *)(coeff + 20));
    in[6] = _mm_load_si128((const __m128i *)(coeff + 24));
    in[7] = _mm_load_si128((const __m128i *)(coeff + 28));
    in[8] = _mm_load_si128((const __m128i *)(coeff + 32));
    in[9] = _mm_load_si128((const __m128i *)(coeff + 36));
    in[10] = _mm_load_si128((const __m128i *)(coeff + 40));
    in[11] = _mm_load_si128((const __m128i *)(coeff + 44));
    in[12] = _mm_load_si128((const __m128i *)(coeff + 48));
    in[13] = _mm_load_si128((const __m128i *)(coeff + 52));
    in[14] = _mm_load_si128((const __m128i *)(coeff + 56));
    in[15] = _mm_load_si128((const __m128i *)(coeff + 60));
}

static void idct8x8_sse4_1(__m128i *in, __m128i *out, int32_t bit) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospim8 = _mm_set1_epi32(-cospi[8]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    __m128i u0, u1, u2, u3, u4, u5, u6, u7;
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;
    __m128i x, y;
    int32_t col;

    // Note:
    //  Even column: 0, 2, ..., 14
    //  Odd column: 1, 3, ..., 15
    //  one even column plus one odd column constructs one row (8 coeffs)
    //  total we have 8 rows (8x8).
    for (col = 0; col < 2; ++col) {
        // stage 0
        // stage 1
        // stage 2
        u0 = in[0 * 2 + col];
        u1 = in[4 * 2 + col];
        u2 = in[2 * 2 + col];
        u3 = in[6 * 2 + col];

        x = _mm_mullo_epi32(in[1 * 2 + col], cospi56);
        y = _mm_mullo_epi32(in[7 * 2 + col], cospim8);
        u4 = _mm_add_epi32(x, y);
        u4 = _mm_add_epi32(u4, rnding);
        u4 = _mm_srai_epi32(u4, bit);

        x = _mm_mullo_epi32(in[1 * 2 + col], cospi8);
        y = _mm_mullo_epi32(in[7 * 2 + col], cospi56);
        u7 = _mm_add_epi32(x, y);
        u7 = _mm_add_epi32(u7, rnding);
        u7 = _mm_srai_epi32(u7, bit);

        x = _mm_mullo_epi32(in[5 * 2 + col], cospi24);
        y = _mm_mullo_epi32(in[3 * 2 + col], cospim40);
        u5 = _mm_add_epi32(x, y);
        u5 = _mm_add_epi32(u5, rnding);
        u5 = _mm_srai_epi32(u5, bit);

        x = _mm_mullo_epi32(in[5 * 2 + col], cospi40);
        y = _mm_mullo_epi32(in[3 * 2 + col], cospi24);
        u6 = _mm_add_epi32(x, y);
        u6 = _mm_add_epi32(u6, rnding);
        u6 = _mm_srai_epi32(u6, bit);

        // stage 3
        x = _mm_mullo_epi32(u0, cospi32);
        y = _mm_mullo_epi32(u1, cospi32);
        v0 = _mm_add_epi32(x, y);
        v0 = _mm_add_epi32(v0, rnding);
        v0 = _mm_srai_epi32(v0, bit);

        v1 = _mm_sub_epi32(x, y);
        v1 = _mm_add_epi32(v1, rnding);
        v1 = _mm_srai_epi32(v1, bit);

        x = _mm_mullo_epi32(u2, cospi48);
        y = _mm_mullo_epi32(u3, cospim16);
        v2 = _mm_add_epi32(x, y);
        v2 = _mm_add_epi32(v2, rnding);
        v2 = _mm_srai_epi32(v2, bit);

        x = _mm_mullo_epi32(u2, cospi16);
        y = _mm_mullo_epi32(u3, cospi48);
        v3 = _mm_add_epi32(x, y);
        v3 = _mm_add_epi32(v3, rnding);
        v3 = _mm_srai_epi32(v3, bit);

        v4 = _mm_add_epi32(u4, u5);
        v5 = _mm_sub_epi32(u4, u5);
        v6 = _mm_sub_epi32(u7, u6);
        v7 = _mm_add_epi32(u6, u7);

        // stage 4
        u0 = _mm_add_epi32(v0, v3);
        u1 = _mm_add_epi32(v1, v2);
        u2 = _mm_sub_epi32(v1, v2);
        u3 = _mm_sub_epi32(v0, v3);
        u4 = v4;
        u7 = v7;

        x = _mm_mullo_epi32(v5, cospi32);
        y = _mm_mullo_epi32(v6, cospi32);
        u6 = _mm_add_epi32(y, x);
        u6 = _mm_add_epi32(u6, rnding);
        u6 = _mm_srai_epi32(u6, bit);

        u5 = _mm_sub_epi32(y, x);
        u5 = _mm_add_epi32(u5, rnding);
        u5 = _mm_srai_epi32(u5, bit);

        // stage 5
        out[0 * 2 + col] = _mm_add_epi32(u0, u7);
        out[1 * 2 + col] = _mm_add_epi32(u1, u6);
        out[2 * 2 + col] = _mm_add_epi32(u2, u5);
        out[3 * 2 + col] = _mm_add_epi32(u3, u4);
        out[4 * 2 + col] = _mm_sub_epi32(u3, u4);
        out[5 * 2 + col] = _mm_sub_epi32(u2, u5);
        out[6 * 2 + col] = _mm_sub_epi32(u1, u6);
        out[7 * 2 + col] = _mm_sub_epi32(u0, u7);
    }
}

static void iadst8x8_sse4_1(__m128i *in, __m128i *out, int32_t bit) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi36 = _mm_set1_epi32(cospi[36]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospi52 = _mm_set1_epi32(cospi[52]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const __m128i kZero = _mm_setzero_si128();
    __m128i u[8], v[8], x;

    // Even 8 points: 0, 2, ..., 14
    // stage 0
    // stage 1
    // stage 2
    // (1)
    u[0] = _mm_mullo_epi32(in[14], cospi4);
    x = _mm_mullo_epi32(in[0], cospi60);
    u[0] = _mm_add_epi32(u[0], x);
    u[0] = _mm_add_epi32(u[0], rnding);
    u[0] = _mm_srai_epi32(u[0], bit);

    u[1] = _mm_mullo_epi32(in[14], cospi60);
    x = _mm_mullo_epi32(in[0], cospi4);
    u[1] = _mm_sub_epi32(u[1], x);
    u[1] = _mm_add_epi32(u[1], rnding);
    u[1] = _mm_srai_epi32(u[1], bit);

    // (2)
    u[2] = _mm_mullo_epi32(in[10], cospi20);
    x = _mm_mullo_epi32(in[4], cospi44);
    u[2] = _mm_add_epi32(u[2], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_mullo_epi32(in[10], cospi44);
    x = _mm_mullo_epi32(in[4], cospi20);
    u[3] = _mm_sub_epi32(u[3], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    // (3)
    u[4] = _mm_mullo_epi32(in[6], cospi36);
    x = _mm_mullo_epi32(in[8], cospi28);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(in[6], cospi28);
    x = _mm_mullo_epi32(in[8], cospi36);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    // (4)
    u[6] = _mm_mullo_epi32(in[2], cospi52);
    x = _mm_mullo_epi32(in[12], cospi12);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(in[2], cospi12);
    x = _mm_mullo_epi32(in[12], cospi52);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 3
    v[0] = _mm_add_epi32(u[0], u[4]);
    v[4] = _mm_sub_epi32(u[0], u[4]);
    v[1] = _mm_add_epi32(u[1], u[5]);
    v[5] = _mm_sub_epi32(u[1], u[5]);
    v[2] = _mm_add_epi32(u[2], u[6]);
    v[6] = _mm_sub_epi32(u[2], u[6]);
    v[3] = _mm_add_epi32(u[3], u[7]);
    v[7] = _mm_sub_epi32(u[3], u[7]);

    // stage 4
    u[0] = v[0];
    u[1] = v[1];
    u[2] = v[2];
    u[3] = v[3];

    u[4] = _mm_mullo_epi32(v[4], cospi16);
    x = _mm_mullo_epi32(v[5], cospi48);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(v[4], cospi48);
    x = _mm_mullo_epi32(v[5], cospi16);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    u[6] = _mm_mullo_epi32(v[6], cospim48);
    x = _mm_mullo_epi32(v[7], cospi16);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(v[6], cospi16);
    x = _mm_mullo_epi32(v[7], cospim48);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 5
    v[0] = _mm_add_epi32(u[0], u[2]);
    v[2] = _mm_sub_epi32(u[0], u[2]);
    v[1] = _mm_add_epi32(u[1], u[3]);
    v[3] = _mm_sub_epi32(u[1], u[3]);
    v[4] = _mm_add_epi32(u[4], u[6]);
    v[6] = _mm_sub_epi32(u[4], u[6]);
    v[5] = _mm_add_epi32(u[5], u[7]);
    v[7] = _mm_sub_epi32(u[5], u[7]);

    // stage 6
    u[0] = v[0];
    u[1] = v[1];
    u[4] = v[4];
    u[5] = v[5];

    v[0] = _mm_mullo_epi32(v[2], cospi32);
    x = _mm_mullo_epi32(v[3], cospi32);
    u[2] = _mm_add_epi32(v[0], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_sub_epi32(v[0], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    v[0] = _mm_mullo_epi32(v[6], cospi32);
    x = _mm_mullo_epi32(v[7], cospi32);
    u[6] = _mm_add_epi32(v[0], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_sub_epi32(v[0], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 7
    out[0] = u[0];
    out[2] = _mm_sub_epi32(kZero, u[4]);
    out[4] = u[6];
    out[6] = _mm_sub_epi32(kZero, u[2]);
    out[8] = u[3];
    out[10] = _mm_sub_epi32(kZero, u[7]);
    out[12] = u[5];
    out[14] = _mm_sub_epi32(kZero, u[1]);

    // Odd 8 points: 1, 3, ..., 15
    // stage 0
    // stage 1
    // stage 2
    // (1)
    u[0] = _mm_mullo_epi32(in[15], cospi4);
    x = _mm_mullo_epi32(in[1], cospi60);
    u[0] = _mm_add_epi32(u[0], x);
    u[0] = _mm_add_epi32(u[0], rnding);
    u[0] = _mm_srai_epi32(u[0], bit);

    u[1] = _mm_mullo_epi32(in[15], cospi60);
    x = _mm_mullo_epi32(in[1], cospi4);
    u[1] = _mm_sub_epi32(u[1], x);
    u[1] = _mm_add_epi32(u[1], rnding);
    u[1] = _mm_srai_epi32(u[1], bit);

    // (2)
    u[2] = _mm_mullo_epi32(in[11], cospi20);
    x = _mm_mullo_epi32(in[5], cospi44);
    u[2] = _mm_add_epi32(u[2], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_mullo_epi32(in[11], cospi44);
    x = _mm_mullo_epi32(in[5], cospi20);
    u[3] = _mm_sub_epi32(u[3], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    // (3)
    u[4] = _mm_mullo_epi32(in[7], cospi36);
    x = _mm_mullo_epi32(in[9], cospi28);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(in[7], cospi28);
    x = _mm_mullo_epi32(in[9], cospi36);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    // (4)
    u[6] = _mm_mullo_epi32(in[3], cospi52);
    x = _mm_mullo_epi32(in[13], cospi12);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(in[3], cospi12);
    x = _mm_mullo_epi32(in[13], cospi52);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 3
    v[0] = _mm_add_epi32(u[0], u[4]);
    v[4] = _mm_sub_epi32(u[0], u[4]);
    v[1] = _mm_add_epi32(u[1], u[5]);
    v[5] = _mm_sub_epi32(u[1], u[5]);
    v[2] = _mm_add_epi32(u[2], u[6]);
    v[6] = _mm_sub_epi32(u[2], u[6]);
    v[3] = _mm_add_epi32(u[3], u[7]);
    v[7] = _mm_sub_epi32(u[3], u[7]);

    // stage 4
    u[0] = v[0];
    u[1] = v[1];
    u[2] = v[2];
    u[3] = v[3];

    u[4] = _mm_mullo_epi32(v[4], cospi16);
    x = _mm_mullo_epi32(v[5], cospi48);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(v[4], cospi48);
    x = _mm_mullo_epi32(v[5], cospi16);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    u[6] = _mm_mullo_epi32(v[6], cospim48);
    x = _mm_mullo_epi32(v[7], cospi16);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(v[6], cospi16);
    x = _mm_mullo_epi32(v[7], cospim48);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 5
    v[0] = _mm_add_epi32(u[0], u[2]);
    v[2] = _mm_sub_epi32(u[0], u[2]);
    v[1] = _mm_add_epi32(u[1], u[3]);
    v[3] = _mm_sub_epi32(u[1], u[3]);
    v[4] = _mm_add_epi32(u[4], u[6]);
    v[6] = _mm_sub_epi32(u[4], u[6]);
    v[5] = _mm_add_epi32(u[5], u[7]);
    v[7] = _mm_sub_epi32(u[5], u[7]);

    // stage 6
    u[0] = v[0];
    u[1] = v[1];
    u[4] = v[4];
    u[5] = v[5];

    v[0] = _mm_mullo_epi32(v[2], cospi32);
    x = _mm_mullo_epi32(v[3], cospi32);
    u[2] = _mm_add_epi32(v[0], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_sub_epi32(v[0], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    v[0] = _mm_mullo_epi32(v[6], cospi32);
    x = _mm_mullo_epi32(v[7], cospi32);
    u[6] = _mm_add_epi32(v[0], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_sub_epi32(v[0], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 7
    out[1] = u[0];
    out[3] = _mm_sub_epi32(kZero, u[4]);
    out[5] = u[6];
    out[7] = _mm_sub_epi32(kZero, u[2]);
    out[9] = u[3];
    out[11] = _mm_sub_epi32(kZero, u[7]);
    out[13] = u[5];
    out[15] = _mm_sub_epi32(kZero, u[1]);
}

static void round_shift_8x8(__m128i *in, int32_t shift) {
    round_shift_4x4(&in[0], shift);
    round_shift_4x4(&in[4], shift);
    round_shift_4x4(&in[8], shift);
    round_shift_4x4(&in[12], shift);
}

static __m128i get_recon_8x8(const __m128i pred, __m128i res_lo, __m128i res_hi,
    int32_t fliplr, int32_t bd) {
    __m128i x0, x1;
    const __m128i zero = _mm_setzero_si128();

    x0 = _mm_unpacklo_epi16(pred, zero);
    x1 = _mm_unpackhi_epi16(pred, zero);

    if (fliplr) {
        res_lo = _mm_shuffle_epi32(res_lo, 0x1B);
        res_hi = _mm_shuffle_epi32(res_hi, 0x1B);
        x0 = _mm_add_epi32(res_hi, x0);
        x1 = _mm_add_epi32(res_lo, x1);
    }
    else {
        x0 = _mm_add_epi32(res_lo, x0);
        x1 = _mm_add_epi32(res_hi, x1);
    }

    x0 = _mm_packus_epi32(x0, x1);
    return highbd_clamp_epi16(x0, bd);
}

static void write_buffer_8x8(__m128i *in, uint16_t *output, int32_t stride,
    int32_t fliplr, int32_t flipud, int32_t shift, int32_t bd) {
    __m128i u0, u1, u2, u3, u4, u5, u6, u7;
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;

    round_shift_8x8(in, shift);

    v0 = _mm_load_si128((__m128i const *)(output + 0 * stride));
    v1 = _mm_load_si128((__m128i const *)(output + 1 * stride));
    v2 = _mm_load_si128((__m128i const *)(output + 2 * stride));
    v3 = _mm_load_si128((__m128i const *)(output + 3 * stride));
    v4 = _mm_load_si128((__m128i const *)(output + 4 * stride));
    v5 = _mm_load_si128((__m128i const *)(output + 5 * stride));
    v6 = _mm_load_si128((__m128i const *)(output + 6 * stride));
    v7 = _mm_load_si128((__m128i const *)(output + 7 * stride));

    if (flipud) {
        u0 = get_recon_8x8(v0, in[14], in[15], fliplr, bd);
        u1 = get_recon_8x8(v1, in[12], in[13], fliplr, bd);
        u2 = get_recon_8x8(v2, in[10], in[11], fliplr, bd);
        u3 = get_recon_8x8(v3, in[8], in[9], fliplr, bd);
        u4 = get_recon_8x8(v4, in[6], in[7], fliplr, bd);
        u5 = get_recon_8x8(v5, in[4], in[5], fliplr, bd);
        u6 = get_recon_8x8(v6, in[2], in[3], fliplr, bd);
        u7 = get_recon_8x8(v7, in[0], in[1], fliplr, bd);
    }
    else {
        u0 = get_recon_8x8(v0, in[0], in[1], fliplr, bd);
        u1 = get_recon_8x8(v1, in[2], in[3], fliplr, bd);
        u2 = get_recon_8x8(v2, in[4], in[5], fliplr, bd);
        u3 = get_recon_8x8(v3, in[6], in[7], fliplr, bd);
        u4 = get_recon_8x8(v4, in[8], in[9], fliplr, bd);
        u5 = get_recon_8x8(v5, in[10], in[11], fliplr, bd);
        u6 = get_recon_8x8(v6, in[12], in[13], fliplr, bd);
        u7 = get_recon_8x8(v7, in[14], in[15], fliplr, bd);
    }

    _mm_store_si128((__m128i *)(output + 0 * stride), u0);
    _mm_store_si128((__m128i *)(output + 1 * stride), u1);
    _mm_store_si128((__m128i *)(output + 2 * stride), u2);
    _mm_store_si128((__m128i *)(output + 3 * stride), u3);
    _mm_store_si128((__m128i *)(output + 4 * stride), u4);
    _mm_store_si128((__m128i *)(output + 5 * stride), u5);
    _mm_store_si128((__m128i *)(output + 6 * stride), u6);
    _mm_store_si128((__m128i *)(output + 7 * stride), u7);
}

void av1_inv_txfm2d_add_8x8_sse4_1(const int32_t *coeff, uint16_t *output,
    int32_t stride, TxType tx_type, int32_t bd) {
    __m128i in[16], out[16];
    const int8_t *shift = inv_txfm_shift_ls[TX_8X8];
    const int32_t txw_idx = get_txw_idx(TX_8X8);
    const int32_t txh_idx = get_txh_idx(TX_8X8);

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        idct8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        idct8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case DCT_ADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        idct8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_DCT:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        idct8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_ADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case FLIPADST_DCT:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        idct8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 1, -shift[1], bd);
        break;
    case DCT_FLIPADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        idct8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case ADST_FLIPADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 1, 1, -shift[1], bd);
        break;
    case FLIPADST_ADST:
        load_buffer_8x8(coeff, in);
        transpose_8x8(in, out);
        iadst8x8_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        transpose_8x8(in, out);
        round_shift_8x8(out, -shift[0]);
        iadst8x8_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_8x8(in, output, stride, 0, 1, -shift[1], bd);
        break;
    default: assert(0);
    }
}

// 16x16
static void load_buffer_16x16(const int32_t *coeff, __m128i *in) {
    int32_t i;
    for (i = 0; i < 64; ++i)
        in[i] = _mm_load_si128((const __m128i *)(coeff + (i << 2)));
}

static void assign_8x8_input_from_16x16(const __m128i *in, __m128i *in8x8,
    int32_t col) {
    int32_t i;
    for (i = 0; i < 16; i += 2) {
        in8x8[i] = in[col];
        in8x8[i + 1] = in[col + 1];
        col += 4;
    }
}

static void swap_addr(uint16_t **output1, uint16_t **output2) {
    uint16_t *tmp;
    tmp = *output1;
    *output1 = *output2;
    *output2 = tmp;
}

static void write_buffer_16x16(__m128i *in, uint16_t *output, int32_t stride,
    int32_t fliplr, int32_t flipud, int32_t shift, int32_t bd) {
    __m128i in8x8[16];
    uint16_t *leftUp = &output[0];
    uint16_t *rightUp = &output[8];
    uint16_t *leftDown = &output[8 * stride];
    uint16_t *rightDown = &output[8 * stride + 8];

    if (fliplr) {
        swap_addr(&leftUp, &rightUp);
        swap_addr(&leftDown, &rightDown);
    }

    if (flipud) {
        swap_addr(&leftUp, &leftDown);
        swap_addr(&rightUp, &rightDown);
    }

    // Left-up quarter
    assign_8x8_input_from_16x16(in, in8x8, 0);
    write_buffer_8x8(in8x8, leftUp, stride, fliplr, flipud, shift, bd);

    // Right-up quarter
    assign_8x8_input_from_16x16(in, in8x8, 2);
    write_buffer_8x8(in8x8, rightUp, stride, fliplr, flipud, shift, bd);

    // Left-down quarter
    assign_8x8_input_from_16x16(in, in8x8, 32);
    write_buffer_8x8(in8x8, leftDown, stride, fliplr, flipud, shift, bd);

    // Right-down quarter
    assign_8x8_input_from_16x16(in, in8x8, 34);
    write_buffer_8x8(in8x8, rightDown, stride, fliplr, flipud, shift, bd);
}

static void idct16x16_sse4_1(__m128i *in, __m128i *out, int32_t bit) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospim4 = _mm_set1_epi32(-cospi[4]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospim36 = _mm_set1_epi32(-cospi[36]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospim20 = _mm_set1_epi32(-cospi[20]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospim52 = _mm_set1_epi32(-cospi[52]);
    const __m128i cospi52 = _mm_set1_epi32(cospi[52]);
    const __m128i cospi36 = _mm_set1_epi32(cospi[36]);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospim8 = _mm_set1_epi32(-cospi[8]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    __m128i u[16], v[16], x, y;
    int32_t col;

    for (col = 0; col < 4; ++col) {
        // stage 0
        // stage 1
        u[0] = in[0 * 4 + col];
        u[1] = in[8 * 4 + col];
        u[2] = in[4 * 4 + col];
        u[3] = in[12 * 4 + col];
        u[4] = in[2 * 4 + col];
        u[5] = in[10 * 4 + col];
        u[6] = in[6 * 4 + col];
        u[7] = in[14 * 4 + col];
        u[8] = in[1 * 4 + col];
        u[9] = in[9 * 4 + col];
        u[10] = in[5 * 4 + col];
        u[11] = in[13 * 4 + col];
        u[12] = in[3 * 4 + col];
        u[13] = in[11 * 4 + col];
        u[14] = in[7 * 4 + col];
        u[15] = in[15 * 4 + col];

        // stage 2
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];
        v[4] = u[4];
        v[5] = u[5];
        v[6] = u[6];
        v[7] = u[7];

        v[8] = half_btf_sse4_1(&cospi60, &u[8], &cospim4, &u[15], &rnding, bit);
        v[9] = half_btf_sse4_1(&cospi28, &u[9], &cospim36, &u[14], &rnding, bit);
        v[10] = half_btf_sse4_1(&cospi44, &u[10], &cospim20, &u[13], &rnding, bit);
        v[11] = half_btf_sse4_1(&cospi12, &u[11], &cospim52, &u[12], &rnding, bit);
        v[12] = half_btf_sse4_1(&cospi52, &u[11], &cospi12, &u[12], &rnding, bit);
        v[13] = half_btf_sse4_1(&cospi20, &u[10], &cospi44, &u[13], &rnding, bit);
        v[14] = half_btf_sse4_1(&cospi36, &u[9], &cospi28, &u[14], &rnding, bit);
        v[15] = half_btf_sse4_1(&cospi4, &u[8], &cospi60, &u[15], &rnding, bit);

        // stage 3
        u[0] = v[0];
        u[1] = v[1];
        u[2] = v[2];
        u[3] = v[3];
        u[4] = half_btf_sse4_1(&cospi56, &v[4], &cospim8, &v[7], &rnding, bit);
        u[5] = half_btf_sse4_1(&cospi24, &v[5], &cospim40, &v[6], &rnding, bit);
        u[6] = half_btf_sse4_1(&cospi40, &v[5], &cospi24, &v[6], &rnding, bit);
        u[7] = half_btf_sse4_1(&cospi8, &v[4], &cospi56, &v[7], &rnding, bit);
        u[8] = _mm_add_epi32(v[8], v[9]);
        u[9] = _mm_sub_epi32(v[8], v[9]);
        u[10] = _mm_sub_epi32(v[11], v[10]);
        u[11] = _mm_add_epi32(v[10], v[11]);
        u[12] = _mm_add_epi32(v[12], v[13]);
        u[13] = _mm_sub_epi32(v[12], v[13]);
        u[14] = _mm_sub_epi32(v[15], v[14]);
        u[15] = _mm_add_epi32(v[14], v[15]);

        // stage 4
        x = _mm_mullo_epi32(u[0], cospi32);
        y = _mm_mullo_epi32(u[1], cospi32);
        v[0] = _mm_add_epi32(x, y);
        v[0] = _mm_add_epi32(v[0], rnding);
        v[0] = _mm_srai_epi32(v[0], bit);

        v[1] = _mm_sub_epi32(x, y);
        v[1] = _mm_add_epi32(v[1], rnding);
        v[1] = _mm_srai_epi32(v[1], bit);

        v[2] = half_btf_sse4_1(&cospi48, &u[2], &cospim16, &u[3], &rnding, bit);
        v[3] = half_btf_sse4_1(&cospi16, &u[2], &cospi48, &u[3], &rnding, bit);
        v[4] = _mm_add_epi32(u[4], u[5]);
        v[5] = _mm_sub_epi32(u[4], u[5]);
        v[6] = _mm_sub_epi32(u[7], u[6]);
        v[7] = _mm_add_epi32(u[6], u[7]);
        v[8] = u[8];
        v[9] = half_btf_sse4_1(&cospim16, &u[9], &cospi48, &u[14], &rnding, bit);
        v[10] = half_btf_sse4_1(&cospim48, &u[10], &cospim16, &u[13], &rnding, bit);
        v[11] = u[11];
        v[12] = u[12];
        v[13] = half_btf_sse4_1(&cospim16, &u[10], &cospi48, &u[13], &rnding, bit);
        v[14] = half_btf_sse4_1(&cospi48, &u[9], &cospi16, &u[14], &rnding, bit);
        v[15] = u[15];

        // stage 5
        u[0] = _mm_add_epi32(v[0], v[3]);
        u[1] = _mm_add_epi32(v[1], v[2]);
        u[2] = _mm_sub_epi32(v[1], v[2]);
        u[3] = _mm_sub_epi32(v[0], v[3]);
        u[4] = v[4];

        x = _mm_mullo_epi32(v[5], cospi32);
        y = _mm_mullo_epi32(v[6], cospi32);
        u[5] = _mm_sub_epi32(y, x);
        u[5] = _mm_add_epi32(u[5], rnding);
        u[5] = _mm_srai_epi32(u[5], bit);

        u[6] = _mm_add_epi32(y, x);
        u[6] = _mm_add_epi32(u[6], rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        u[7] = v[7];
        u[8] = _mm_add_epi32(v[8], v[11]);
        u[9] = _mm_add_epi32(v[9], v[10]);
        u[10] = _mm_sub_epi32(v[9], v[10]);
        u[11] = _mm_sub_epi32(v[8], v[11]);
        u[12] = _mm_sub_epi32(v[15], v[12]);
        u[13] = _mm_sub_epi32(v[14], v[13]);
        u[14] = _mm_add_epi32(v[13], v[14]);
        u[15] = _mm_add_epi32(v[12], v[15]);

        // stage 6
        v[0] = _mm_add_epi32(u[0], u[7]);
        v[1] = _mm_add_epi32(u[1], u[6]);
        v[2] = _mm_add_epi32(u[2], u[5]);
        v[3] = _mm_add_epi32(u[3], u[4]);
        v[4] = _mm_sub_epi32(u[3], u[4]);
        v[5] = _mm_sub_epi32(u[2], u[5]);
        v[6] = _mm_sub_epi32(u[1], u[6]);
        v[7] = _mm_sub_epi32(u[0], u[7]);
        v[8] = u[8];
        v[9] = u[9];

        x = _mm_mullo_epi32(u[10], cospi32);
        y = _mm_mullo_epi32(u[13], cospi32);
        v[10] = _mm_sub_epi32(y, x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[13] = _mm_add_epi32(x, y);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        x = _mm_mullo_epi32(u[11], cospi32);
        y = _mm_mullo_epi32(u[12], cospi32);
        v[11] = _mm_sub_epi32(y, x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_add_epi32(x, y);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[14] = u[14];
        v[15] = u[15];

        // stage 7
        out[0 * 4 + col] = _mm_add_epi32(v[0], v[15]);
        out[1 * 4 + col] = _mm_add_epi32(v[1], v[14]);
        out[2 * 4 + col] = _mm_add_epi32(v[2], v[13]);
        out[3 * 4 + col] = _mm_add_epi32(v[3], v[12]);
        out[4 * 4 + col] = _mm_add_epi32(v[4], v[11]);
        out[5 * 4 + col] = _mm_add_epi32(v[5], v[10]);
        out[6 * 4 + col] = _mm_add_epi32(v[6], v[9]);
        out[7 * 4 + col] = _mm_add_epi32(v[7], v[8]);
        out[8 * 4 + col] = _mm_sub_epi32(v[7], v[8]);
        out[9 * 4 + col] = _mm_sub_epi32(v[6], v[9]);
        out[10 * 4 + col] = _mm_sub_epi32(v[5], v[10]);
        out[11 * 4 + col] = _mm_sub_epi32(v[4], v[11]);
        out[12 * 4 + col] = _mm_sub_epi32(v[3], v[12]);
        out[13 * 4 + col] = _mm_sub_epi32(v[2], v[13]);
        out[14 * 4 + col] = _mm_sub_epi32(v[1], v[14]);
        out[15 * 4 + col] = _mm_sub_epi32(v[0], v[15]);
    }
}

static void iadst16x16_sse4_1(__m128i *in, __m128i *out, int32_t bit) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi2 = _mm_set1_epi32(cospi[2]);
    const __m128i cospi62 = _mm_set1_epi32(cospi[62]);
    const __m128i cospi10 = _mm_set1_epi32(cospi[10]);
    const __m128i cospi54 = _mm_set1_epi32(cospi[54]);
    const __m128i cospi18 = _mm_set1_epi32(cospi[18]);
    const __m128i cospi46 = _mm_set1_epi32(cospi[46]);
    const __m128i cospi26 = _mm_set1_epi32(cospi[26]);
    const __m128i cospi38 = _mm_set1_epi32(cospi[38]);
    const __m128i cospi34 = _mm_set1_epi32(cospi[34]);
    const __m128i cospi30 = _mm_set1_epi32(cospi[30]);
    const __m128i cospi42 = _mm_set1_epi32(cospi[42]);
    const __m128i cospi22 = _mm_set1_epi32(cospi[22]);
    const __m128i cospi50 = _mm_set1_epi32(cospi[50]);
    const __m128i cospi14 = _mm_set1_epi32(cospi[14]);
    const __m128i cospi58 = _mm_set1_epi32(cospi[58]);
    const __m128i cospi6 = _mm_set1_epi32(cospi[6]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim56 = _mm_set1_epi32(-cospi[56]);
    const __m128i cospim24 = _mm_set1_epi32(-cospi[24]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    __m128i u[16], v[16], x, y;
    const int32_t col_num = 4;
    int32_t col;

    // Calculate the column 0, 1, 2, 3
    for (col = 0; col < col_num; ++col) {
        // stage 0
        // stage 1
        // stage 2
        v[0] = _mm_mullo_epi32(in[15 * col_num + col], cospi2);
        x = _mm_mullo_epi32(in[0 * col_num + col], cospi62);
        v[0] = _mm_add_epi32(v[0], x);
        v[0] = _mm_add_epi32(v[0], rnding);
        v[0] = _mm_srai_epi32(v[0], bit);

        v[1] = _mm_mullo_epi32(in[15 * col_num + col], cospi62);
        x = _mm_mullo_epi32(in[0 * col_num + col], cospi2);
        v[1] = _mm_sub_epi32(v[1], x);
        v[1] = _mm_add_epi32(v[1], rnding);
        v[1] = _mm_srai_epi32(v[1], bit);

        v[2] = _mm_mullo_epi32(in[13 * col_num + col], cospi10);
        x = _mm_mullo_epi32(in[2 * col_num + col], cospi54);
        v[2] = _mm_add_epi32(v[2], x);
        v[2] = _mm_add_epi32(v[2], rnding);
        v[2] = _mm_srai_epi32(v[2], bit);

        v[3] = _mm_mullo_epi32(in[13 * col_num + col], cospi54);
        x = _mm_mullo_epi32(in[2 * col_num + col], cospi10);
        v[3] = _mm_sub_epi32(v[3], x);
        v[3] = _mm_add_epi32(v[3], rnding);
        v[3] = _mm_srai_epi32(v[3], bit);

        v[4] = _mm_mullo_epi32(in[11 * col_num + col], cospi18);
        x = _mm_mullo_epi32(in[4 * col_num + col], cospi46);
        v[4] = _mm_add_epi32(v[4], x);
        v[4] = _mm_add_epi32(v[4], rnding);
        v[4] = _mm_srai_epi32(v[4], bit);

        v[5] = _mm_mullo_epi32(in[11 * col_num + col], cospi46);
        x = _mm_mullo_epi32(in[4 * col_num + col], cospi18);
        v[5] = _mm_sub_epi32(v[5], x);
        v[5] = _mm_add_epi32(v[5], rnding);
        v[5] = _mm_srai_epi32(v[5], bit);

        v[6] = _mm_mullo_epi32(in[9 * col_num + col], cospi26);
        x = _mm_mullo_epi32(in[6 * col_num + col], cospi38);
        v[6] = _mm_add_epi32(v[6], x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_mullo_epi32(in[9 * col_num + col], cospi38);
        x = _mm_mullo_epi32(in[6 * col_num + col], cospi26);
        v[7] = _mm_sub_epi32(v[7], x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = _mm_mullo_epi32(in[7 * col_num + col], cospi34);
        x = _mm_mullo_epi32(in[8 * col_num + col], cospi30);
        v[8] = _mm_add_epi32(v[8], x);
        v[8] = _mm_add_epi32(v[8], rnding);
        v[8] = _mm_srai_epi32(v[8], bit);

        v[9] = _mm_mullo_epi32(in[7 * col_num + col], cospi30);
        x = _mm_mullo_epi32(in[8 * col_num + col], cospi34);
        v[9] = _mm_sub_epi32(v[9], x);
        v[9] = _mm_add_epi32(v[9], rnding);
        v[9] = _mm_srai_epi32(v[9], bit);

        v[10] = _mm_mullo_epi32(in[5 * col_num + col], cospi42);
        x = _mm_mullo_epi32(in[10 * col_num + col], cospi22);
        v[10] = _mm_add_epi32(v[10], x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_mullo_epi32(in[5 * col_num + col], cospi22);
        x = _mm_mullo_epi32(in[10 * col_num + col], cospi42);
        v[11] = _mm_sub_epi32(v[11], x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_mullo_epi32(in[3 * col_num + col], cospi50);
        x = _mm_mullo_epi32(in[12 * col_num + col], cospi14);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(in[3 * col_num + col], cospi14);
        x = _mm_mullo_epi32(in[12 * col_num + col], cospi50);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(in[1 * col_num + col], cospi58);
        x = _mm_mullo_epi32(in[14 * col_num + col], cospi6);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(in[1 * col_num + col], cospi6);
        x = _mm_mullo_epi32(in[14 * col_num + col], cospi58);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 3
        u[0] = _mm_add_epi32(v[0], v[8]);
        u[8] = _mm_sub_epi32(v[0], v[8]);
        u[1] = _mm_add_epi32(v[1], v[9]);
        u[9] = _mm_sub_epi32(v[1], v[9]);
        u[2] = _mm_add_epi32(v[2], v[10]);
        u[10] = _mm_sub_epi32(v[2], v[10]);
        u[3] = _mm_add_epi32(v[3], v[11]);
        u[11] = _mm_sub_epi32(v[3], v[11]);
        u[4] = _mm_add_epi32(v[4], v[12]);
        u[12] = _mm_sub_epi32(v[4], v[12]);
        u[5] = _mm_add_epi32(v[5], v[13]);
        u[13] = _mm_sub_epi32(v[5], v[13]);
        u[6] = _mm_add_epi32(v[6], v[14]);
        u[14] = _mm_sub_epi32(v[6], v[14]);
        u[7] = _mm_add_epi32(v[7], v[15]);
        u[15] = _mm_sub_epi32(v[7], v[15]);

        // stage 4
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];
        v[4] = u[4];
        v[5] = u[5];
        v[6] = u[6];
        v[7] = u[7];

        v[8] = _mm_mullo_epi32(u[8], cospi8);
        x = _mm_mullo_epi32(u[9], cospi56);
        v[8] = _mm_add_epi32(v[8], x);
        v[8] = _mm_add_epi32(v[8], rnding);
        v[8] = _mm_srai_epi32(v[8], bit);

        v[9] = _mm_mullo_epi32(u[8], cospi56);
        x = _mm_mullo_epi32(u[9], cospi8);
        v[9] = _mm_sub_epi32(v[9], x);
        v[9] = _mm_add_epi32(v[9], rnding);
        v[9] = _mm_srai_epi32(v[9], bit);

        v[10] = _mm_mullo_epi32(u[10], cospi40);
        x = _mm_mullo_epi32(u[11], cospi24);
        v[10] = _mm_add_epi32(v[10], x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_mullo_epi32(u[10], cospi24);
        x = _mm_mullo_epi32(u[11], cospi40);
        v[11] = _mm_sub_epi32(v[11], x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_mullo_epi32(u[12], cospim56);
        x = _mm_mullo_epi32(u[13], cospi8);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(u[12], cospi8);
        x = _mm_mullo_epi32(u[13], cospim56);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(u[14], cospim24);
        x = _mm_mullo_epi32(u[15], cospi40);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(u[14], cospi40);
        x = _mm_mullo_epi32(u[15], cospim24);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 5
        u[0] = _mm_add_epi32(v[0], v[4]);
        u[4] = _mm_sub_epi32(v[0], v[4]);
        u[1] = _mm_add_epi32(v[1], v[5]);
        u[5] = _mm_sub_epi32(v[1], v[5]);
        u[2] = _mm_add_epi32(v[2], v[6]);
        u[6] = _mm_sub_epi32(v[2], v[6]);
        u[3] = _mm_add_epi32(v[3], v[7]);
        u[7] = _mm_sub_epi32(v[3], v[7]);
        u[8] = _mm_add_epi32(v[8], v[12]);
        u[12] = _mm_sub_epi32(v[8], v[12]);
        u[9] = _mm_add_epi32(v[9], v[13]);
        u[13] = _mm_sub_epi32(v[9], v[13]);
        u[10] = _mm_add_epi32(v[10], v[14]);
        u[14] = _mm_sub_epi32(v[10], v[14]);
        u[11] = _mm_add_epi32(v[11], v[15]);
        u[15] = _mm_sub_epi32(v[11], v[15]);

        // stage 6
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];

        v[4] = _mm_mullo_epi32(u[4], cospi16);
        x = _mm_mullo_epi32(u[5], cospi48);
        v[4] = _mm_add_epi32(v[4], x);
        v[4] = _mm_add_epi32(v[4], rnding);
        v[4] = _mm_srai_epi32(v[4], bit);

        v[5] = _mm_mullo_epi32(u[4], cospi48);
        x = _mm_mullo_epi32(u[5], cospi16);
        v[5] = _mm_sub_epi32(v[5], x);
        v[5] = _mm_add_epi32(v[5], rnding);
        v[5] = _mm_srai_epi32(v[5], bit);

        v[6] = _mm_mullo_epi32(u[6], cospim48);
        x = _mm_mullo_epi32(u[7], cospi16);
        v[6] = _mm_add_epi32(v[6], x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_mullo_epi32(u[6], cospi16);
        x = _mm_mullo_epi32(u[7], cospim48);
        v[7] = _mm_sub_epi32(v[7], x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = u[8];
        v[9] = u[9];
        v[10] = u[10];
        v[11] = u[11];

        v[12] = _mm_mullo_epi32(u[12], cospi16);
        x = _mm_mullo_epi32(u[13], cospi48);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(u[12], cospi48);
        x = _mm_mullo_epi32(u[13], cospi16);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(u[14], cospim48);
        x = _mm_mullo_epi32(u[15], cospi16);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(u[14], cospi16);
        x = _mm_mullo_epi32(u[15], cospim48);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 7
        u[0] = _mm_add_epi32(v[0], v[2]);
        u[2] = _mm_sub_epi32(v[0], v[2]);
        u[1] = _mm_add_epi32(v[1], v[3]);
        u[3] = _mm_sub_epi32(v[1], v[3]);
        u[4] = _mm_add_epi32(v[4], v[6]);
        u[6] = _mm_sub_epi32(v[4], v[6]);
        u[5] = _mm_add_epi32(v[5], v[7]);
        u[7] = _mm_sub_epi32(v[5], v[7]);
        u[8] = _mm_add_epi32(v[8], v[10]);
        u[10] = _mm_sub_epi32(v[8], v[10]);
        u[9] = _mm_add_epi32(v[9], v[11]);
        u[11] = _mm_sub_epi32(v[9], v[11]);
        u[12] = _mm_add_epi32(v[12], v[14]);
        u[14] = _mm_sub_epi32(v[12], v[14]);
        u[13] = _mm_add_epi32(v[13], v[15]);
        u[15] = _mm_sub_epi32(v[13], v[15]);

        // stage 8
        v[0] = u[0];
        v[1] = u[1];

        y = _mm_mullo_epi32(u[2], cospi32);
        x = _mm_mullo_epi32(u[3], cospi32);
        v[2] = _mm_add_epi32(y, x);
        v[2] = _mm_add_epi32(v[2], rnding);
        v[2] = _mm_srai_epi32(v[2], bit);

        v[3] = _mm_sub_epi32(y, x);
        v[3] = _mm_add_epi32(v[3], rnding);
        v[3] = _mm_srai_epi32(v[3], bit);

        v[4] = u[4];
        v[5] = u[5];

        y = _mm_mullo_epi32(u[6], cospi32);
        x = _mm_mullo_epi32(u[7], cospi32);
        v[6] = _mm_add_epi32(y, x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_sub_epi32(y, x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = u[8];
        v[9] = u[9];

        y = _mm_mullo_epi32(u[10], cospi32);
        x = _mm_mullo_epi32(u[11], cospi32);
        v[10] = _mm_add_epi32(y, x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_sub_epi32(y, x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = u[12];
        v[13] = u[13];

        y = _mm_mullo_epi32(u[14], cospi32);
        x = _mm_mullo_epi32(u[15], cospi32);
        v[14] = _mm_add_epi32(y, x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_sub_epi32(y, x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 9
        out[0 * col_num + col] = v[0];
        out[1 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[8]);
        out[2 * col_num + col] = v[12];
        out[3 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[4]);
        out[4 * col_num + col] = v[6];
        out[5 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[14]);
        out[6 * col_num + col] = v[10];
        out[7 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[2]);
        out[8 * col_num + col] = v[3];
        out[9 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[11]);
        out[10 * col_num + col] = v[15];
        out[11 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[7]);
        out[12 * col_num + col] = v[5];
        out[13 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[13]);
        out[14 * col_num + col] = v[9];
        out[15 * col_num + col] = _mm_sub_epi32(_mm_setzero_si128(), v[1]);
    }
}

static void round_shift_16x16(__m128i *in, int32_t shift) {
    round_shift_8x8(&in[0], shift);
    round_shift_8x8(&in[16], shift);
    round_shift_8x8(&in[32], shift);
    round_shift_8x8(&in[48], shift);
}

void av1_inv_txfm2d_add_16x16_sse4_1(const int32_t *coeff, uint16_t *output,
    int32_t stride, TxType tx_type, int32_t bd) {
    __m128i in[64], out[64];
    const int8_t *shift = inv_txfm_shift_ls[TX_16X16];
    const int32_t txw_idx = get_txw_idx(TX_16X16);
    const int32_t txh_idx = get_txh_idx(TX_16X16);

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case DCT_ADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_DCT:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case ADST_ADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 0, -shift[1], bd);
        break;
    case FLIPADST_DCT:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 1, -shift[1], bd);
        break;
    case DCT_FLIPADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        idct16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case ADST_FLIPADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 1, 0, -shift[1], bd);
        break;
    case FLIPADST_FLIPADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 1, 1, -shift[1], bd);
        break;
    case FLIPADST_ADST:
        load_buffer_16x16(coeff, in);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx]);
        round_shift_16x16(in, -shift[0]);
        transpose_16x16(in, out);
        iadst16x16_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx]);
        write_buffer_16x16(in, output, stride, 0, 1, -shift[1], bd);
        break;
    default: assert(0);
    }
}

static void load_buffer_64x64_lower_32x32(const int32_t *coeff, __m128i *in) {
    int32_t i, j;

    __m128i zero = _mm_setzero_si128();

    for (i = 0; i < 32; ++i) {
        for (j = 0; j < 8; ++j) {
            in[16 * i + j] =
                _mm_loadu_si128((const __m128i *)(coeff + 32 * i + 4 * j));
            in[16 * i + j + 8] = zero;
        }
    }

    for (i = 0; i < 512; ++i) in[512 + i] = zero;
}

static void transpose_64x64(__m128i *in, __m128i *out, int32_t do_cols) {
    int32_t i, j;
    for (i = 0; i < (do_cols ? 16 : 8); ++i) {
        for (j = 0; j < 8; ++j) {
            TRANSPOSE_4X4(in[(4 * i + 0) * 16 + j], in[(4 * i + 1) * 16 + j],
                in[(4 * i + 2) * 16 + j], in[(4 * i + 3) * 16 + j],
                out[(4 * j + 0) * 16 + i], out[(4 * j + 1) * 16 + i],
                out[(4 * j + 2) * 16 + i], out[(4 * j + 3) * 16 + i]);
        }
    }
}

static void round_shift_32x32(__m128i *in, int32_t shift) {
    round_shift_16x16(&in[0], shift);
    round_shift_16x16(&in[64], shift);
    round_shift_16x16(&in[128], shift);
    round_shift_16x16(&in[192], shift);
}

static void round_shift_64x64(__m128i *in, int32_t shift) {
    round_shift_32x32(&in[0], shift);
    round_shift_32x32(&in[256], shift);
}

static void assign_16x16_input_from_32x32(const __m128i *in, __m128i *in16x16,
    int32_t col) {
    int32_t i;
    for (i = 0; i < 16 * 16 / 4; i += 4) {
        in16x16[i] = in[col];
        in16x16[i + 1] = in[col + 1];
        in16x16[i + 2] = in[col + 2];
        in16x16[i + 3] = in[col + 3];
        col += 8;
    }
}

static void write_buffer_32x32(__m128i *in, uint16_t *output, int32_t stride,
    int32_t fliplr, int32_t flipud, int32_t shift, int32_t bd) {
    __m128i in16x16[16 * 16 / 4];
    uint16_t *leftUp = &output[0];
    uint16_t *rightUp = &output[16];
    uint16_t *leftDown = &output[16 * stride];
    uint16_t *rightDown = &output[16 * stride + 16];

    if (fliplr) {
        swap_addr(&leftUp, &rightUp);
        swap_addr(&leftDown, &rightDown);
    }

    if (flipud) {
        swap_addr(&leftUp, &leftDown);
        swap_addr(&rightUp, &rightDown);
    }

    // Left-up quarter
    assign_16x16_input_from_32x32(in, in16x16, 0);
    write_buffer_16x16(in16x16, leftUp, stride, fliplr, flipud, shift, bd);

    // Right-up quarter
    assign_16x16_input_from_32x32(in, in16x16, 32 / 2 / 4);
    write_buffer_16x16(in16x16, rightUp, stride, fliplr, flipud, shift, bd);

    // Left-down quarter
    assign_16x16_input_from_32x32(in, in16x16, 32 * 32 / 2 / 4);
    write_buffer_16x16(in16x16, leftDown, stride, fliplr, flipud, shift, bd);

    // Right-down quarter
    assign_16x16_input_from_32x32(in, in16x16, 32 * 32 / 2 / 4 + 32 / 2 / 4);
    write_buffer_16x16(in16x16, rightDown, stride, fliplr, flipud, shift, bd);
}

static void assign_32x32_input_from_64x64(const __m128i *in, __m128i *in32x32,
    int32_t col) {
    int32_t i;
    for (i = 0; i < 32 * 32 / 4; i += 8) {
        in32x32[i] = in[col];
        in32x32[i + 1] = in[col + 1];
        in32x32[i + 2] = in[col + 2];
        in32x32[i + 3] = in[col + 3];
        in32x32[i + 4] = in[col + 4];
        in32x32[i + 5] = in[col + 5];
        in32x32[i + 6] = in[col + 6];
        in32x32[i + 7] = in[col + 7];
        col += 16;
    }
}

static void write_buffer_64x64(__m128i *in, uint16_t *output, int32_t stride,
    int32_t fliplr, int32_t flipud, int32_t shift, int32_t bd) {
    __m128i in32x32[32 * 32 / 4];
    uint16_t *leftUp = &output[0];
    uint16_t *rightUp = &output[32];
    uint16_t *leftDown = &output[32 * stride];
    uint16_t *rightDown = &output[32 * stride + 32];

    if (fliplr) {
        swap_addr(&leftUp, &rightUp);
        swap_addr(&leftDown, &rightDown);
    }

    if (flipud) {
        swap_addr(&leftUp, &leftDown);
        swap_addr(&rightUp, &rightDown);
    }

    // Left-up quarter
    assign_32x32_input_from_64x64(in, in32x32, 0);
    write_buffer_32x32(in32x32, leftUp, stride, fliplr, flipud, shift, bd);

    // Right-up quarter
    assign_32x32_input_from_64x64(in, in32x32, 64 / 2 / 4);
    write_buffer_32x32(in32x32, rightUp, stride, fliplr, flipud, shift, bd);

    // Left-down quarter
    assign_32x32_input_from_64x64(in, in32x32, 64 * 64 / 2 / 4);
    write_buffer_32x32(in32x32, leftDown, stride, fliplr, flipud, shift, bd);

    // Right-down quarter
    assign_32x32_input_from_64x64(in, in32x32, 64 * 64 / 2 / 4 + 64 / 2 / 4);
    write_buffer_32x32(in32x32, rightDown, stride, fliplr, flipud, shift, bd);
}

static void idct64x64_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd) {
    int32_t i, j;
    const int32_t *cospi = cospi_arr(bit);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    int32_t col;

    const __m128i cospi1 = _mm_set1_epi32(cospi[1]);
    const __m128i cospi2 = _mm_set1_epi32(cospi[2]);
    const __m128i cospi3 = _mm_set1_epi32(cospi[3]);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi5 = _mm_set1_epi32(cospi[5]);
    const __m128i cospi6 = _mm_set1_epi32(cospi[6]);
    const __m128i cospi7 = _mm_set1_epi32(cospi[7]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi9 = _mm_set1_epi32(cospi[9]);
    const __m128i cospi10 = _mm_set1_epi32(cospi[10]);
    const __m128i cospi11 = _mm_set1_epi32(cospi[11]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospi13 = _mm_set1_epi32(cospi[13]);
    const __m128i cospi14 = _mm_set1_epi32(cospi[14]);
    const __m128i cospi15 = _mm_set1_epi32(cospi[15]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospi17 = _mm_set1_epi32(cospi[17]);
    const __m128i cospi18 = _mm_set1_epi32(cospi[18]);
    const __m128i cospi19 = _mm_set1_epi32(cospi[19]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospi21 = _mm_set1_epi32(cospi[21]);
    const __m128i cospi22 = _mm_set1_epi32(cospi[22]);
    const __m128i cospi23 = _mm_set1_epi32(cospi[23]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospi25 = _mm_set1_epi32(cospi[25]);
    const __m128i cospi26 = _mm_set1_epi32(cospi[26]);
    const __m128i cospi27 = _mm_set1_epi32(cospi[27]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospi29 = _mm_set1_epi32(cospi[29]);
    const __m128i cospi30 = _mm_set1_epi32(cospi[30]);
    const __m128i cospi31 = _mm_set1_epi32(cospi[31]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi35 = _mm_set1_epi32(cospi[35]);
    const __m128i cospi36 = _mm_set1_epi32(cospi[36]);
    const __m128i cospi38 = _mm_set1_epi32(cospi[38]);
    const __m128i cospi39 = _mm_set1_epi32(cospi[39]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi43 = _mm_set1_epi32(cospi[43]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi46 = _mm_set1_epi32(cospi[46]);
    const __m128i cospi47 = _mm_set1_epi32(cospi[47]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi51 = _mm_set1_epi32(cospi[51]);
    const __m128i cospi52 = _mm_set1_epi32(cospi[52]);
    const __m128i cospi54 = _mm_set1_epi32(cospi[54]);
    const __m128i cospi55 = _mm_set1_epi32(cospi[55]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi59 = _mm_set1_epi32(cospi[59]);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospi62 = _mm_set1_epi32(cospi[62]);
    const __m128i cospi63 = _mm_set1_epi32(cospi[63]);

    const __m128i cospim4 = _mm_set1_epi32(-cospi[4]);
    const __m128i cospim8 = _mm_set1_epi32(-cospi[8]);
    const __m128i cospim12 = _mm_set1_epi32(-cospi[12]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospim20 = _mm_set1_epi32(-cospi[20]);
    const __m128i cospim24 = _mm_set1_epi32(-cospi[24]);
    const __m128i cospim28 = _mm_set1_epi32(-cospi[28]);
    const __m128i cospim32 = _mm_set1_epi32(-cospi[32]);
    const __m128i cospim33 = _mm_set1_epi32(-cospi[33]);
    const __m128i cospim34 = _mm_set1_epi32(-cospi[34]);
    const __m128i cospim36 = _mm_set1_epi32(-cospi[36]);
    const __m128i cospim37 = _mm_set1_epi32(-cospi[37]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospim41 = _mm_set1_epi32(-cospi[41]);
    const __m128i cospim42 = _mm_set1_epi32(-cospi[42]);
    const __m128i cospim44 = _mm_set1_epi32(-cospi[44]);
    const __m128i cospim45 = _mm_set1_epi32(-cospi[45]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospim49 = _mm_set1_epi32(-cospi[49]);
    const __m128i cospim50 = _mm_set1_epi32(-cospi[50]);
    const __m128i cospim52 = _mm_set1_epi32(-cospi[52]);
    const __m128i cospim53 = _mm_set1_epi32(-cospi[53]);
    const __m128i cospim56 = _mm_set1_epi32(-cospi[56]);
    const __m128i cospim57 = _mm_set1_epi32(-cospi[57]);
    const __m128i cospim58 = _mm_set1_epi32(-cospi[58]);
    const __m128i cospim60 = _mm_set1_epi32(-cospi[60]);
    const __m128i cospim61 = _mm_set1_epi32(-cospi[61]);

    for (col = 0; col < (do_cols ? 64 / 4 : 32 / 4); ++col) {
        __m128i u[64], v[64];

        // stage 1
        u[32] = in[1 * 16 + col];
        u[34] = in[17 * 16 + col];
        u[36] = in[9 * 16 + col];
        u[38] = in[25 * 16 + col];
        u[40] = in[5 * 16 + col];
        u[42] = in[21 * 16 + col];
        u[44] = in[13 * 16 + col];
        u[46] = in[29 * 16 + col];
        u[48] = in[3 * 16 + col];
        u[50] = in[19 * 16 + col];
        u[52] = in[11 * 16 + col];
        u[54] = in[27 * 16 + col];
        u[56] = in[7 * 16 + col];
        u[58] = in[23 * 16 + col];
        u[60] = in[15 * 16 + col];
        u[62] = in[31 * 16 + col];

        v[16] = in[2 * 16 + col];
        v[18] = in[18 * 16 + col];
        v[20] = in[10 * 16 + col];
        v[22] = in[26 * 16 + col];
        v[24] = in[6 * 16 + col];
        v[26] = in[22 * 16 + col];
        v[28] = in[14 * 16 + col];
        v[30] = in[30 * 16 + col];

        u[8] = in[4 * 16 + col];
        u[10] = in[20 * 16 + col];
        u[12] = in[12 * 16 + col];
        u[14] = in[28 * 16 + col];

        v[4] = in[8 * 16 + col];
        v[6] = in[24 * 16 + col];

        u[0] = in[0 * 16 + col];
        u[2] = in[16 * 16 + col];

        // stage 2
        v[32] = half_btf_0_sse4_1(&cospi63, &u[32], &rnding, bit);
        v[33] = half_btf_0_sse4_1(&cospim33, &u[62], &rnding, bit);
        v[34] = half_btf_0_sse4_1(&cospi47, &u[34], &rnding, bit);
        v[35] = half_btf_0_sse4_1(&cospim49, &u[60], &rnding, bit);
        v[36] = half_btf_0_sse4_1(&cospi55, &u[36], &rnding, bit);
        v[37] = half_btf_0_sse4_1(&cospim41, &u[58], &rnding, bit);
        v[38] = half_btf_0_sse4_1(&cospi39, &u[38], &rnding, bit);
        v[39] = half_btf_0_sse4_1(&cospim57, &u[56], &rnding, bit);
        v[40] = half_btf_0_sse4_1(&cospi59, &u[40], &rnding, bit);
        v[41] = half_btf_0_sse4_1(&cospim37, &u[54], &rnding, bit);
        v[42] = half_btf_0_sse4_1(&cospi43, &u[42], &rnding, bit);
        v[43] = half_btf_0_sse4_1(&cospim53, &u[52], &rnding, bit);
        v[44] = half_btf_0_sse4_1(&cospi51, &u[44], &rnding, bit);
        v[45] = half_btf_0_sse4_1(&cospim45, &u[50], &rnding, bit);
        v[46] = half_btf_0_sse4_1(&cospi35, &u[46], &rnding, bit);
        v[47] = half_btf_0_sse4_1(&cospim61, &u[48], &rnding, bit);
        v[48] = half_btf_0_sse4_1(&cospi3, &u[48], &rnding, bit);
        v[49] = half_btf_0_sse4_1(&cospi29, &u[46], &rnding, bit);
        v[50] = half_btf_0_sse4_1(&cospi19, &u[50], &rnding, bit);
        v[51] = half_btf_0_sse4_1(&cospi13, &u[44], &rnding, bit);
        v[52] = half_btf_0_sse4_1(&cospi11, &u[52], &rnding, bit);
        v[53] = half_btf_0_sse4_1(&cospi21, &u[42], &rnding, bit);
        v[54] = half_btf_0_sse4_1(&cospi27, &u[54], &rnding, bit);
        v[55] = half_btf_0_sse4_1(&cospi5, &u[40], &rnding, bit);
        v[56] = half_btf_0_sse4_1(&cospi7, &u[56], &rnding, bit);
        v[57] = half_btf_0_sse4_1(&cospi25, &u[38], &rnding, bit);
        v[58] = half_btf_0_sse4_1(&cospi23, &u[58], &rnding, bit);
        v[59] = half_btf_0_sse4_1(&cospi9, &u[36], &rnding, bit);
        v[60] = half_btf_0_sse4_1(&cospi15, &u[60], &rnding, bit);
        v[61] = half_btf_0_sse4_1(&cospi17, &u[34], &rnding, bit);
        v[62] = half_btf_0_sse4_1(&cospi31, &u[62], &rnding, bit);
        v[63] = half_btf_0_sse4_1(&cospi1, &u[32], &rnding, bit);

        // stage 3
        u[16] = half_btf_0_sse4_1(&cospi62, &v[16], &rnding, bit);
        u[17] = half_btf_0_sse4_1(&cospim34, &v[30], &rnding, bit);
        u[18] = half_btf_0_sse4_1(&cospi46, &v[18], &rnding, bit);
        u[19] = half_btf_0_sse4_1(&cospim50, &v[28], &rnding, bit);
        u[20] = half_btf_0_sse4_1(&cospi54, &v[20], &rnding, bit);
        u[21] = half_btf_0_sse4_1(&cospim42, &v[26], &rnding, bit);
        u[22] = half_btf_0_sse4_1(&cospi38, &v[22], &rnding, bit);
        u[23] = half_btf_0_sse4_1(&cospim58, &v[24], &rnding, bit);
        u[24] = half_btf_0_sse4_1(&cospi6, &v[24], &rnding, bit);
        u[25] = half_btf_0_sse4_1(&cospi26, &v[22], &rnding, bit);
        u[26] = half_btf_0_sse4_1(&cospi22, &v[26], &rnding, bit);
        u[27] = half_btf_0_sse4_1(&cospi10, &v[20], &rnding, bit);
        u[28] = half_btf_0_sse4_1(&cospi14, &v[28], &rnding, bit);
        u[29] = half_btf_0_sse4_1(&cospi18, &v[18], &rnding, bit);
        u[30] = half_btf_0_sse4_1(&cospi30, &v[30], &rnding, bit);
        u[31] = half_btf_0_sse4_1(&cospi2, &v[16], &rnding, bit);

        for (i = 32; i < 64; i += 4) {
            addsub_sse4_1(v[i + 0], v[i + 1], &u[i + 0], &u[i + 1], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(v[i + 3], v[i + 2], &u[i + 3], &u[i + 2], &clamp_lo,
                &clamp_hi);
        }

        // stage 4
        v[8] = half_btf_0_sse4_1(&cospi60, &u[8], &rnding, bit);
        v[9] = half_btf_0_sse4_1(&cospim36, &u[14], &rnding, bit);
        v[10] = half_btf_0_sse4_1(&cospi44, &u[10], &rnding, bit);
        v[11] = half_btf_0_sse4_1(&cospim52, &u[12], &rnding, bit);
        v[12] = half_btf_0_sse4_1(&cospi12, &u[12], &rnding, bit);
        v[13] = half_btf_0_sse4_1(&cospi20, &u[10], &rnding, bit);
        v[14] = half_btf_0_sse4_1(&cospi28, &u[14], &rnding, bit);
        v[15] = half_btf_0_sse4_1(&cospi4, &u[8], &rnding, bit);

        for (i = 16; i < 32; i += 4) {
            addsub_sse4_1(u[i + 0], u[i + 1], &v[i + 0], &v[i + 1], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(u[i + 3], u[i + 2], &v[i + 3], &v[i + 2], &clamp_lo,
                &clamp_hi);
        }

        for (i = 32; i < 64; i += 4) {
            v[i + 0] = u[i + 0];
            v[i + 3] = u[i + 3];
        }

        v[33] = half_btf_sse4_1(&cospim4, &u[33], &cospi60, &u[62], &rnding, bit);
        v[34] = half_btf_sse4_1(&cospim60, &u[34], &cospim4, &u[61], &rnding, bit);
        v[37] = half_btf_sse4_1(&cospim36, &u[37], &cospi28, &u[58], &rnding, bit);
        v[38] = half_btf_sse4_1(&cospim28, &u[38], &cospim36, &u[57], &rnding, bit);
        v[41] = half_btf_sse4_1(&cospim20, &u[41], &cospi44, &u[54], &rnding, bit);
        v[42] = half_btf_sse4_1(&cospim44, &u[42], &cospim20, &u[53], &rnding, bit);
        v[45] = half_btf_sse4_1(&cospim52, &u[45], &cospi12, &u[50], &rnding, bit);
        v[46] = half_btf_sse4_1(&cospim12, &u[46], &cospim52, &u[49], &rnding, bit);
        v[49] = half_btf_sse4_1(&cospim52, &u[46], &cospi12, &u[49], &rnding, bit);
        v[50] = half_btf_sse4_1(&cospi12, &u[45], &cospi52, &u[50], &rnding, bit);
        v[53] = half_btf_sse4_1(&cospim20, &u[42], &cospi44, &u[53], &rnding, bit);
        v[54] = half_btf_sse4_1(&cospi44, &u[41], &cospi20, &u[54], &rnding, bit);
        v[57] = half_btf_sse4_1(&cospim36, &u[38], &cospi28, &u[57], &rnding, bit);
        v[58] = half_btf_sse4_1(&cospi28, &u[37], &cospi36, &u[58], &rnding, bit);
        v[61] = half_btf_sse4_1(&cospim4, &u[34], &cospi60, &u[61], &rnding, bit);
        v[62] = half_btf_sse4_1(&cospi60, &u[33], &cospi4, &u[62], &rnding, bit);

        // stage 5
        u[4] = half_btf_0_sse4_1(&cospi56, &v[4], &rnding, bit);
        u[5] = half_btf_0_sse4_1(&cospim40, &v[6], &rnding, bit);
        u[6] = half_btf_0_sse4_1(&cospi24, &v[6], &rnding, bit);
        u[7] = half_btf_0_sse4_1(&cospi8, &v[4], &rnding, bit);

        for (i = 8; i < 16; i += 4) {
            addsub_sse4_1(v[i + 0], v[i + 1], &u[i + 0], &u[i + 1], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(v[i + 3], v[i + 2], &u[i + 3], &u[i + 2], &clamp_lo,
                &clamp_hi);
        }

        for (i = 16; i < 32; i += 4) {
            u[i + 0] = v[i + 0];
            u[i + 3] = v[i + 3];
        }

        u[17] = half_btf_sse4_1(&cospim8, &v[17], &cospi56, &v[30], &rnding, bit);
        u[18] = half_btf_sse4_1(&cospim56, &v[18], &cospim8, &v[29], &rnding, bit);
        u[21] = half_btf_sse4_1(&cospim40, &v[21], &cospi24, &v[26], &rnding, bit);
        u[22] = half_btf_sse4_1(&cospim24, &v[22], &cospim40, &v[25], &rnding, bit);
        u[25] = half_btf_sse4_1(&cospim40, &v[22], &cospi24, &v[25], &rnding, bit);
        u[26] = half_btf_sse4_1(&cospi24, &v[21], &cospi40, &v[26], &rnding, bit);
        u[29] = half_btf_sse4_1(&cospim8, &v[18], &cospi56, &v[29], &rnding, bit);
        u[30] = half_btf_sse4_1(&cospi56, &v[17], &cospi8, &v[30], &rnding, bit);

        for (i = 32; i < 64; i += 8) {
            addsub_sse4_1(v[i + 0], v[i + 3], &u[i + 0], &u[i + 3], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(v[i + 1], v[i + 2], &u[i + 1], &u[i + 2], &clamp_lo,
                &clamp_hi);

            addsub_sse4_1(v[i + 7], v[i + 4], &u[i + 7], &u[i + 4], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(v[i + 6], v[i + 5], &u[i + 6], &u[i + 5], &clamp_lo,
                &clamp_hi);
        }

        // stage 6
        v[0] = half_btf_0_sse4_1(&cospi32, &u[0], &rnding, bit);
        v[1] = half_btf_0_sse4_1(&cospi32, &u[0], &rnding, bit);
        v[2] = half_btf_0_sse4_1(&cospi48, &u[2], &rnding, bit);
        v[3] = half_btf_0_sse4_1(&cospi16, &u[2], &rnding, bit);

        addsub_sse4_1(u[4], u[5], &v[4], &v[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[7], u[6], &v[7], &v[6], &clamp_lo, &clamp_hi);

        for (i = 8; i < 16; i += 4) {
            v[i + 0] = u[i + 0];
            v[i + 3] = u[i + 3];
        }

        v[9] = half_btf_sse4_1(&cospim16, &u[9], &cospi48, &u[14], &rnding, bit);
        v[10] = half_btf_sse4_1(&cospim48, &u[10], &cospim16, &u[13], &rnding, bit);
        v[13] = half_btf_sse4_1(&cospim16, &u[10], &cospi48, &u[13], &rnding, bit);
        v[14] = half_btf_sse4_1(&cospi48, &u[9], &cospi16, &u[14], &rnding, bit);

        for (i = 16; i < 32; i += 8) {
            addsub_sse4_1(u[i + 0], u[i + 3], &v[i + 0], &v[i + 3], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(u[i + 1], u[i + 2], &v[i + 1], &v[i + 2], &clamp_lo,
                &clamp_hi);

            addsub_sse4_1(u[i + 7], u[i + 4], &v[i + 7], &v[i + 4], &clamp_lo,
                &clamp_hi);
            addsub_sse4_1(u[i + 6], u[i + 5], &v[i + 6], &v[i + 5], &clamp_lo,
                &clamp_hi);
        }

        for (i = 32; i < 64; i += 8) {
            v[i + 0] = u[i + 0];
            v[i + 1] = u[i + 1];
            v[i + 6] = u[i + 6];
            v[i + 7] = u[i + 7];
        }

        v[34] = half_btf_sse4_1(&cospim8, &u[34], &cospi56, &u[61], &rnding, bit);
        v[35] = half_btf_sse4_1(&cospim8, &u[35], &cospi56, &u[60], &rnding, bit);
        v[36] = half_btf_sse4_1(&cospim56, &u[36], &cospim8, &u[59], &rnding, bit);
        v[37] = half_btf_sse4_1(&cospim56, &u[37], &cospim8, &u[58], &rnding, bit);
        v[42] = half_btf_sse4_1(&cospim40, &u[42], &cospi24, &u[53], &rnding, bit);
        v[43] = half_btf_sse4_1(&cospim40, &u[43], &cospi24, &u[52], &rnding, bit);
        v[44] = half_btf_sse4_1(&cospim24, &u[44], &cospim40, &u[51], &rnding, bit);
        v[45] = half_btf_sse4_1(&cospim24, &u[45], &cospim40, &u[50], &rnding, bit);
        v[50] = half_btf_sse4_1(&cospim40, &u[45], &cospi24, &u[50], &rnding, bit);
        v[51] = half_btf_sse4_1(&cospim40, &u[44], &cospi24, &u[51], &rnding, bit);
        v[52] = half_btf_sse4_1(&cospi24, &u[43], &cospi40, &u[52], &rnding, bit);
        v[53] = half_btf_sse4_1(&cospi24, &u[42], &cospi40, &u[53], &rnding, bit);
        v[58] = half_btf_sse4_1(&cospim8, &u[37], &cospi56, &u[58], &rnding, bit);
        v[59] = half_btf_sse4_1(&cospim8, &u[36], &cospi56, &u[59], &rnding, bit);
        v[60] = half_btf_sse4_1(&cospi56, &u[35], &cospi8, &u[60], &rnding, bit);
        v[61] = half_btf_sse4_1(&cospi56, &u[34], &cospi8, &u[61], &rnding, bit);

        // stage 7
        addsub_sse4_1(v[0], v[3], &u[0], &u[3], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[1], v[2], &u[1], &u[2], &clamp_lo, &clamp_hi);

        u[4] = v[4];
        u[7] = v[7];
        u[5] = half_btf_sse4_1(&cospim32, &v[5], &cospi32, &v[6], &rnding, bit);
        u[6] = half_btf_sse4_1(&cospi32, &v[5], &cospi32, &v[6], &rnding, bit);

        addsub_sse4_1(v[8], v[11], &u[8], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[9], v[10], &u[9], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[15], v[12], &u[15], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[14], v[13], &u[14], &u[13], &clamp_lo, &clamp_hi);

        for (i = 16; i < 32; i += 8) {
            u[i + 0] = v[i + 0];
            u[i + 1] = v[i + 1];
            u[i + 6] = v[i + 6];
            u[i + 7] = v[i + 7];
        }

        u[18] = half_btf_sse4_1(&cospim16, &v[18], &cospi48, &v[29], &rnding, bit);
        u[19] = half_btf_sse4_1(&cospim16, &v[19], &cospi48, &v[28], &rnding, bit);
        u[20] = half_btf_sse4_1(&cospim48, &v[20], &cospim16, &v[27], &rnding, bit);
        u[21] = half_btf_sse4_1(&cospim48, &v[21], &cospim16, &v[26], &rnding, bit);
        u[26] = half_btf_sse4_1(&cospim16, &v[21], &cospi48, &v[26], &rnding, bit);
        u[27] = half_btf_sse4_1(&cospim16, &v[20], &cospi48, &v[27], &rnding, bit);
        u[28] = half_btf_sse4_1(&cospi48, &v[19], &cospi16, &v[28], &rnding, bit);
        u[29] = half_btf_sse4_1(&cospi48, &v[18], &cospi16, &v[29], &rnding, bit);

        for (i = 32; i < 64; i += 16) {
            for (j = i; j < i + 4; j++) {
                addsub_sse4_1(v[j], v[j ^ 7], &u[j], &u[j ^ 7], &clamp_lo, &clamp_hi);
                addsub_sse4_1(v[j ^ 15], v[j ^ 8], &u[j ^ 15], &u[j ^ 8], &clamp_lo,
                    &clamp_hi);
            }
        }

        // stage 8
        for (i = 0; i < 4; ++i)
            addsub_sse4_1(u[i], u[7 - i], &v[i], &v[7 - i], &clamp_lo, &clamp_hi);
        v[8] = u[8];
        v[9] = u[9];
        v[14] = u[14];
        v[15] = u[15];

        v[10] = half_btf_sse4_1(&cospim32, &u[10], &cospi32, &u[13], &rnding, bit);
        v[11] = half_btf_sse4_1(&cospim32, &u[11], &cospi32, &u[12], &rnding, bit);
        v[12] = half_btf_sse4_1(&cospi32, &u[11], &cospi32, &u[12], &rnding, bit);
        v[13] = half_btf_sse4_1(&cospi32, &u[10], &cospi32, &u[13], &rnding, bit);

        for (i = 16; i < 20; ++i) {
            addsub_sse4_1(u[i], u[i ^ 7], &v[i], &v[i ^ 7], &clamp_lo, &clamp_hi);
            addsub_sse4_1(u[i ^ 15], u[i ^ 8], &v[i ^ 15], &v[i ^ 8], &clamp_lo,
                &clamp_hi);
        }

        for (i = 32; i < 36; ++i) {
            v[i] = u[i];
            v[i + 12] = u[i + 12];
            v[i + 16] = u[i + 16];
            v[i + 28] = u[i + 28];
        }

        v[36] = half_btf_sse4_1(&cospim16, &u[36], &cospi48, &u[59], &rnding, bit);
        v[37] = half_btf_sse4_1(&cospim16, &u[37], &cospi48, &u[58], &rnding, bit);
        v[38] = half_btf_sse4_1(&cospim16, &u[38], &cospi48, &u[57], &rnding, bit);
        v[39] = half_btf_sse4_1(&cospim16, &u[39], &cospi48, &u[56], &rnding, bit);
        v[40] = half_btf_sse4_1(&cospim48, &u[40], &cospim16, &u[55], &rnding, bit);
        v[41] = half_btf_sse4_1(&cospim48, &u[41], &cospim16, &u[54], &rnding, bit);
        v[42] = half_btf_sse4_1(&cospim48, &u[42], &cospim16, &u[53], &rnding, bit);
        v[43] = half_btf_sse4_1(&cospim48, &u[43], &cospim16, &u[52], &rnding, bit);
        v[52] = half_btf_sse4_1(&cospim16, &u[43], &cospi48, &u[52], &rnding, bit);
        v[53] = half_btf_sse4_1(&cospim16, &u[42], &cospi48, &u[53], &rnding, bit);
        v[54] = half_btf_sse4_1(&cospim16, &u[41], &cospi48, &u[54], &rnding, bit);
        v[55] = half_btf_sse4_1(&cospim16, &u[40], &cospi48, &u[55], &rnding, bit);
        v[56] = half_btf_sse4_1(&cospi48, &u[39], &cospi16, &u[56], &rnding, bit);
        v[57] = half_btf_sse4_1(&cospi48, &u[38], &cospi16, &u[57], &rnding, bit);
        v[58] = half_btf_sse4_1(&cospi48, &u[37], &cospi16, &u[58], &rnding, bit);
        v[59] = half_btf_sse4_1(&cospi48, &u[36], &cospi16, &u[59], &rnding, bit);

        // stage 9
        for (i = 0; i < 8; ++i)
            addsub_sse4_1(v[i], v[15 - i], &u[i], &u[15 - i], &clamp_lo, &clamp_hi);
        for (i = 16; i < 20; ++i) {
            u[i] = v[i];
            u[i + 12] = v[i + 12];
        }

        u[20] = half_btf_sse4_1(&cospim32, &v[20], &cospi32, &v[27], &rnding, bit);
        u[21] = half_btf_sse4_1(&cospim32, &v[21], &cospi32, &v[26], &rnding, bit);
        u[22] = half_btf_sse4_1(&cospim32, &v[22], &cospi32, &v[25], &rnding, bit);
        u[23] = half_btf_sse4_1(&cospim32, &v[23], &cospi32, &v[24], &rnding, bit);
        u[24] = half_btf_sse4_1(&cospi32, &v[23], &cospi32, &v[24], &rnding, bit);
        u[25] = half_btf_sse4_1(&cospi32, &v[22], &cospi32, &v[25], &rnding, bit);
        u[26] = half_btf_sse4_1(&cospi32, &v[21], &cospi32, &v[26], &rnding, bit);
        u[27] = half_btf_sse4_1(&cospi32, &v[20], &cospi32, &v[27], &rnding, bit);

        for (i = 32; i < 40; i++)
            addsub_sse4_1(v[i], v[i ^ 15], &u[i], &u[i ^ 15], &clamp_lo, &clamp_hi);
        for (i = 48; i < 56; i++)
            addsub_sse4_1(v[i ^ 15], v[i], &u[i ^ 15], &u[i], &clamp_lo, &clamp_hi);
        // stage 10
        for (i = 0; i < 16; i++)
            addsub_sse4_1(u[i], u[31 - i], &v[i], &v[31 - i], &clamp_lo, &clamp_hi);
        for (i = 32; i < 40; i++) v[i] = u[i];

        v[40] = half_btf_sse4_1(&cospim32, &u[40], &cospi32, &u[55], &rnding, bit);
        v[41] = half_btf_sse4_1(&cospim32, &u[41], &cospi32, &u[54], &rnding, bit);
        v[42] = half_btf_sse4_1(&cospim32, &u[42], &cospi32, &u[53], &rnding, bit);
        v[43] = half_btf_sse4_1(&cospim32, &u[43], &cospi32, &u[52], &rnding, bit);
        v[44] = half_btf_sse4_1(&cospim32, &u[44], &cospi32, &u[51], &rnding, bit);
        v[45] = half_btf_sse4_1(&cospim32, &u[45], &cospi32, &u[50], &rnding, bit);
        v[46] = half_btf_sse4_1(&cospim32, &u[46], &cospi32, &u[49], &rnding, bit);
        v[47] = half_btf_sse4_1(&cospim32, &u[47], &cospi32, &u[48], &rnding, bit);
        v[48] = half_btf_sse4_1(&cospi32, &u[47], &cospi32, &u[48], &rnding, bit);
        v[49] = half_btf_sse4_1(&cospi32, &u[46], &cospi32, &u[49], &rnding, bit);
        v[50] = half_btf_sse4_1(&cospi32, &u[45], &cospi32, &u[50], &rnding, bit);
        v[51] = half_btf_sse4_1(&cospi32, &u[44], &cospi32, &u[51], &rnding, bit);
        v[52] = half_btf_sse4_1(&cospi32, &u[43], &cospi32, &u[52], &rnding, bit);
        v[53] = half_btf_sse4_1(&cospi32, &u[42], &cospi32, &u[53], &rnding, bit);
        v[54] = half_btf_sse4_1(&cospi32, &u[41], &cospi32, &u[54], &rnding, bit);
        v[55] = half_btf_sse4_1(&cospi32, &u[40], &cospi32, &u[55], &rnding, bit);

        for (i = 56; i < 64; i++) v[i] = u[i];

        // stage 11
        for (i = 0; i < 32; i++) {
            addsub_sse4_1(v[i], v[63 - i], &out[16 * (i)+col],
                &out[16 * (63 - i) + col], &clamp_lo, &clamp_hi);
        }
    }
}

void av1_inv_txfm2d_add_64x64_sse4_1(const int32_t *coeff, uint16_t *output,
    int32_t stride, TxType tx_type, int32_t bd) {
    __m128i in[64 * 64 / 4], out[64 * 64 / 4];
    const int8_t *shift = inv_txfm_shift_ls[TX_64X64];
    const int32_t txw_idx = tx_size_wide_log2[TX_64X64] - tx_size_wide_log2[0];
    const int32_t txh_idx = tx_size_high_log2[TX_64X64] - tx_size_high_log2[0];

    switch (tx_type) {
    case DCT_DCT:
        load_buffer_64x64_lower_32x32(coeff, in);
        transpose_64x64(in, out, 0);
        idct64x64_sse4_1(out, in, inv_cos_bit_row[txw_idx][txh_idx], 0, bd);
        // transpose before shift, so shift can apply to 512 contiguous values
        transpose_64x64(in, out, 1);
        round_shift_64x64(out, -shift[0]);
        idct64x64_sse4_1(out, in, inv_cos_bit_col[txw_idx][txh_idx], 1, bd);
        write_buffer_64x64(in, output, stride, 0, 0, -shift[1], bd);
        break;

    default:
        av1_inv_txfm2d_add_64x64_c(coeff, output, stride, tx_type, bd);
        break;
    }
}

//4x8
static INLINE void load_buffer_32bit_input(const int32_t *in, int32_t stride,
    __m128i *out, int32_t out_size) {
    for (int32_t i = 0; i < out_size; ++i)
        out[i] = _mm_loadu_si128((const __m128i *)(in + i * stride));
}

static void highbd_clamp_epi32_sse4_1(const __m128i *in, __m128i *out,
    const __m128i *clamp_lo,
    const __m128i *clamp_hi, int32_t size) {
    __m128i a0, a1;
    for (int32_t i = 0; i < size; i += 4) {
        a0 = _mm_max_epi32(in[i], *clamp_lo);
        out[i] = _mm_min_epi32(a0, *clamp_hi);

        a1 = _mm_max_epi32(in[i + 1], *clamp_lo);
        out[i + 1] = _mm_min_epi32(a1, *clamp_hi);

        a0 = _mm_max_epi32(in[i + 2], *clamp_lo);
        out[i + 2] = _mm_min_epi32(a0, *clamp_hi);

        a1 = _mm_max_epi32(in[i + 3], *clamp_lo);
        out[i + 3] = _mm_min_epi32(a1, *clamp_hi);
    }
}

static void addsub_shift_sse4_1(const __m128i in0, const __m128i in1,
    __m128i *out0, __m128i *out1,
    const __m128i *clamp_lo,
    const __m128i *clamp_hi, int32_t shift) {
    __m128i offset = _mm_set1_epi32((1 << shift) >> 1);
    __m128i in0_w_offset = _mm_add_epi32(in0, offset);
    __m128i a0 = _mm_add_epi32(in0_w_offset, in1);
    __m128i a1 = _mm_sub_epi32(in0_w_offset, in1);

    a0 = _mm_sra_epi32(a0, _mm_cvtsi32_si128(shift));
    a1 = _mm_sra_epi32(a1, _mm_cvtsi32_si128(shift));

    a0 = _mm_max_epi32(a0, *clamp_lo);
    a0 = _mm_min_epi32(a0, *clamp_hi);
    a1 = _mm_max_epi32(a1, *clamp_lo);
    a1 = _mm_min_epi32(a1, *clamp_hi);

    *out0 = a0;
    *out1 = a1;
}

static void neg_shift_sse4_1(const __m128i in0, const __m128i in1,
    __m128i *out0, __m128i *out1,
    const __m128i *clamp_lo, const __m128i *clamp_hi,
    int32_t shift) {
    __m128i offset = _mm_set1_epi32((1 << shift) >> 1);
    __m128i a0 = _mm_add_epi32(offset, in0);
    __m128i a1 = _mm_sub_epi32(offset, in1);

    a0 = _mm_sra_epi32(a0, _mm_cvtsi32_si128(shift));
    a1 = _mm_sra_epi32(a1, _mm_cvtsi32_si128(shift));

    a0 = _mm_max_epi32(a0, *clamp_lo);
    a0 = _mm_min_epi32(a0, *clamp_hi);
    a1 = _mm_max_epi32(a1, *clamp_lo);
    a1 = _mm_min_epi32(a1, *clamp_hi);

    *out0 = a0;
    *out1 = a1;
}

static void shift_sse4_1(const __m128i *in, __m128i *out,
    const __m128i *clamp_lo, const __m128i *clamp_hi,
    int32_t shift, int32_t size) {
    __m128i offset = _mm_set1_epi32((1 << shift) >> 1);
    __m128i shift_vec = _mm_cvtsi32_si128(shift);
    __m128i a0, a1;
    for (int32_t i = 0; i < size; i += 4) {
        a0 = _mm_add_epi32(in[i], offset);
        a1 = _mm_add_epi32(in[i + 1], offset);
        a0 = _mm_sra_epi32(a0, shift_vec);
        a1 = _mm_sra_epi32(a1, shift_vec);
        a0 = _mm_max_epi32(a0, *clamp_lo);
        a1 = _mm_max_epi32(a1, *clamp_lo);
        out[i] = _mm_min_epi32(a0, *clamp_hi);
        out[i + 1] = _mm_min_epi32(a1, *clamp_hi);

        a0 = _mm_add_epi32(in[i + 2], offset);
        a1 = _mm_add_epi32(in[i + 3], offset);
        a0 = _mm_sra_epi32(a0, shift_vec);
        a1 = _mm_sra_epi32(a1, shift_vec);
        a0 = _mm_max_epi32(a0, *clamp_lo);
        a1 = _mm_max_epi32(a1, *clamp_lo);
        out[i + 2] = _mm_min_epi32(a0, *clamp_hi);
        out[i + 3] = _mm_min_epi32(a1, *clamp_hi);
    }
}

static INLINE __m128i av1_round_shift_32_sse4_1(__m128i vec, int32_t bit) {
    __m128i tmp, round;
    round = _mm_set1_epi32(1 << (bit - 1));
    tmp = _mm_add_epi32(vec, round);
    return _mm_srai_epi32(tmp, bit);
}

static INLINE void av1_round_shift_array_32_sse4_1(__m128i *input,
    __m128i *output,
    const int32_t size,
    const int32_t bit) {
    if (bit > 0) {
        int32_t i;
        for (i = 0; i < size; i++)
            output[i] = av1_round_shift_32_sse4_1(input[i], bit);
    }
    else {
        int32_t i;
        for (i = 0; i < size; i++)
            output[i] = _mm_slli_epi32(input[i], -bit);
    }
}

static INLINE void av1_round_shift_rect_array_32_sse4_1(__m128i *input,
    __m128i *output,
    const int32_t size,
    const int32_t bit,
    const int32_t val) {
    const __m128i sqrt2 = _mm_set1_epi32(val);
    if (bit > 0) {
        int32_t i;
        for (i = 0; i < size; i++) {
            const __m128i r0 = av1_round_shift_32_sse4_1(input[i], bit);
            const __m128i r1 = _mm_mullo_epi32(sqrt2, r0);
            output[i] = av1_round_shift_32_sse4_1(r1, NewSqrt2Bits);
        }
    }
    else {
        int32_t i;
        for (i = 0; i < size; i++) {
            const __m128i r0 = _mm_slli_epi32(input[i], -bit);
            const __m128i r1 = _mm_mullo_epi32(sqrt2, r0);
            output[i] = av1_round_shift_32_sse4_1(r1, NewSqrt2Bits);
        }
    }
}

static void iidentity4_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    (void)bit;
    (void)out_shift;
    __m128i v[4];
    __m128i fact = _mm_set1_epi32(NewSqrt2);
    __m128i offset = _mm_set1_epi32(1 << (NewSqrt2Bits - 1));
    __m128i a0, a1;

    a0 = _mm_mullo_epi32(in[0], fact);
    a1 = _mm_mullo_epi32(in[1], fact);
    a0 = _mm_add_epi32(a0, offset);
    a1 = _mm_add_epi32(a1, offset);
    out[0] = _mm_srai_epi32(a0, NewSqrt2Bits);
    out[1] = _mm_srai_epi32(a1, NewSqrt2Bits);

    a0 = _mm_mullo_epi32(in[2], fact);
    a1 = _mm_mullo_epi32(in[3], fact);
    a0 = _mm_add_epi32(a0, offset);
    a1 = _mm_add_epi32(a1, offset);
    out[2] = _mm_srai_epi32(a0, NewSqrt2Bits);
    out[3] = _mm_srai_epi32(a1, NewSqrt2Bits);

    if (!do_cols) {
        const int32_t log_range = AOMMAX(16, bd + 6);
        const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
        const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);

        highbd_clamp_epi32_sse4_1(out, out, &clamp_lo, &clamp_hi, 4);
    }

    // Transpose for 4x4
    v[0] = _mm_unpacklo_epi32(out[0], out[1]);
    v[1] = _mm_unpackhi_epi32(out[0], out[1]);
    v[2] = _mm_unpacklo_epi32(out[2], out[3]);
    v[3] = _mm_unpackhi_epi32(out[2], out[3]);

    out[0] = _mm_unpacklo_epi64(v[0], v[2]);
    out[1] = _mm_unpackhi_epi64(v[0], v[2]);
    out[2] = _mm_unpacklo_epi64(v[1], v[3]);
    out[3] = _mm_unpackhi_epi64(v[1], v[3]);
}

static void iidentity8_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    (void)bit;
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i v[8];
    v[0] = _mm_add_epi32(in[0], in[0]);
    v[1] = _mm_add_epi32(in[1], in[1]);
    v[2] = _mm_add_epi32(in[2], in[2]);
    v[3] = _mm_add_epi32(in[3], in[3]);
    v[4] = _mm_add_epi32(in[4], in[4]);
    v[5] = _mm_add_epi32(in[5], in[5]);
    v[6] = _mm_add_epi32(in[6], in[6]);
    v[7] = _mm_add_epi32(in[7], in[7]);

    if (!do_cols) {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
            -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
        const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
            (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));

        shift_sse4_1(v, out, &clamp_lo_out, &clamp_hi_out, out_shift, 8);
    }
    else
        highbd_clamp_epi32_sse4_1(v, out, &clamp_lo, &clamp_hi, 8);
}

static void iidentity16_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    (void)bit;
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i v[16];
    __m128i fact = _mm_set1_epi32(2 * NewSqrt2);
    __m128i offset = _mm_set1_epi32(1 << (NewSqrt2Bits - 1));
    __m128i a0, a1, a2, a3;

    for (int32_t i = 0; i < 16; i += 8) {
        a0 = _mm_mullo_epi32(in[i], fact);
        a1 = _mm_mullo_epi32(in[i + 1], fact);
        a0 = _mm_add_epi32(a0, offset);
        a1 = _mm_add_epi32(a1, offset);
        v[i] = _mm_srai_epi32(a0, NewSqrt2Bits);
        v[i + 1] = _mm_srai_epi32(a1, NewSqrt2Bits);

        a2 = _mm_mullo_epi32(in[i + 2], fact);
        a3 = _mm_mullo_epi32(in[i + 3], fact);
        a2 = _mm_add_epi32(a2, offset);
        a3 = _mm_add_epi32(a3, offset);
        v[i + 2] = _mm_srai_epi32(a2, NewSqrt2Bits);
        v[i + 3] = _mm_srai_epi32(a3, NewSqrt2Bits);

        a0 = _mm_mullo_epi32(in[i + 4], fact);
        a1 = _mm_mullo_epi32(in[i + 5], fact);
        a0 = _mm_add_epi32(a0, offset);
        a1 = _mm_add_epi32(a1, offset);
        v[i + 4] = _mm_srai_epi32(a0, NewSqrt2Bits);
        v[i + 5] = _mm_srai_epi32(a1, NewSqrt2Bits);

        a2 = _mm_mullo_epi32(in[i + 6], fact);
        a3 = _mm_mullo_epi32(in[i + 7], fact);
        a2 = _mm_add_epi32(a2, offset);
        a3 = _mm_add_epi32(a3, offset);
        v[i + 6] = _mm_srai_epi32(a2, NewSqrt2Bits);
        v[i + 7] = _mm_srai_epi32(a3, NewSqrt2Bits);
    }

    if (!do_cols) {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
            -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
        const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
            (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));

        shift_sse4_1(v, out, &clamp_lo_out, &clamp_hi_out, out_shift, 16);
    }
    else
        highbd_clamp_epi32_sse4_1(v, out, &clamp_lo, &clamp_hi, 16);
}

static INLINE __m128i highbd_get_recon_4xn_sse4_1(const __m128i pred,
    __m128i res0, const int32_t bd) {
    __m128i x0 = _mm_cvtepi16_epi32(pred);

    x0 = _mm_add_epi32(res0, x0);
    x0 = _mm_packus_epi32(x0, x0);
    x0 = highbd_clamp_epi16(x0, bd);
    return x0;
}

static void idct8x8_new_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospim8 = _mm_set1_epi32(-cospi[8]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u0, u1, u2, u3, u4, u5, u6, u7;
    __m128i v0, v1, v2, v3, v4, v5, v6, v7;
    __m128i x, y;

    // stage 0
    // stage 1
    // stage 2
    u0 = in[0];
    u1 = in[4];
    u2 = in[2];
    u3 = in[6];

    x = _mm_mullo_epi32(in[1], cospi56);
    y = _mm_mullo_epi32(in[7], cospim8);
    u4 = _mm_add_epi32(x, y);
    u4 = _mm_add_epi32(u4, rnding);
    u4 = _mm_srai_epi32(u4, bit);

    x = _mm_mullo_epi32(in[1], cospi8);
    y = _mm_mullo_epi32(in[7], cospi56);
    u7 = _mm_add_epi32(x, y);
    u7 = _mm_add_epi32(u7, rnding);
    u7 = _mm_srai_epi32(u7, bit);

    x = _mm_mullo_epi32(in[5], cospi24);
    y = _mm_mullo_epi32(in[3], cospim40);
    u5 = _mm_add_epi32(x, y);
    u5 = _mm_add_epi32(u5, rnding);
    u5 = _mm_srai_epi32(u5, bit);

    x = _mm_mullo_epi32(in[5], cospi40);
    y = _mm_mullo_epi32(in[3], cospi24);
    u6 = _mm_add_epi32(x, y);
    u6 = _mm_add_epi32(u6, rnding);
    u6 = _mm_srai_epi32(u6, bit);

    // stage 3
    x = _mm_mullo_epi32(u0, cospi32);
    y = _mm_mullo_epi32(u1, cospi32);
    v0 = _mm_add_epi32(x, y);
    v0 = _mm_add_epi32(v0, rnding);
    v0 = _mm_srai_epi32(v0, bit);

    v1 = _mm_sub_epi32(x, y);
    v1 = _mm_add_epi32(v1, rnding);
    v1 = _mm_srai_epi32(v1, bit);

    x = _mm_mullo_epi32(u2, cospi48);
    y = _mm_mullo_epi32(u3, cospim16);
    v2 = _mm_add_epi32(x, y);
    v2 = _mm_add_epi32(v2, rnding);
    v2 = _mm_srai_epi32(v2, bit);

    x = _mm_mullo_epi32(u2, cospi16);
    y = _mm_mullo_epi32(u3, cospi48);
    v3 = _mm_add_epi32(x, y);
    v3 = _mm_add_epi32(v3, rnding);
    v3 = _mm_srai_epi32(v3, bit);

    addsub_sse4_1(u4, u5, &v4, &v5, &clamp_lo, &clamp_hi);
    addsub_sse4_1(u7, u6, &v7, &v6, &clamp_lo, &clamp_hi);

    // stage 4
    addsub_sse4_1(v0, v3, &u0, &u3, &clamp_lo, &clamp_hi);
    addsub_sse4_1(v1, v2, &u1, &u2, &clamp_lo, &clamp_hi);
    u4 = v4;
    u7 = v7;

    x = _mm_mullo_epi32(v5, cospi32);
    y = _mm_mullo_epi32(v6, cospi32);
    u6 = _mm_add_epi32(y, x);
    u6 = _mm_add_epi32(u6, rnding);
    u6 = _mm_srai_epi32(u6, bit);

    u5 = _mm_sub_epi32(y, x);
    u5 = _mm_add_epi32(u5, rnding);
    u5 = _mm_srai_epi32(u5, bit);

    // stage 5
    if (do_cols) {
        addsub_no_clamp_sse4_1(u0, u7, out + 0, out + 7);
        addsub_no_clamp_sse4_1(u1, u6, out + 1, out + 6);
        addsub_no_clamp_sse4_1(u2, u5, out + 2, out + 5);
        addsub_no_clamp_sse4_1(u3, u4, out + 3, out + 4);
    }
    else {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
            -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
        const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
            (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));
        addsub_shift_sse4_1(u0, u7, out + 0, out + 7, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        addsub_shift_sse4_1(u1, u6, out + 1, out + 6, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        addsub_shift_sse4_1(u2, u5, out + 2, out + 5, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        addsub_shift_sse4_1(u3, u4, out + 3, out + 4, &clamp_lo_out, &clamp_hi_out,
            out_shift);
    }
}

static void idct8x8_low1_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    __m128i x;

    // stage 0
    // stage 1
    // stage 2
    // stage 3
    x = _mm_mullo_epi32(in[0], cospi32);
    x = _mm_add_epi32(x, rnding);
    x = _mm_srai_epi32(x, bit);

    // stage 4
    // stage 5
    if (!do_cols) {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
            -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
        const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
            (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));

        __m128i offset = _mm_set1_epi32((1 << out_shift) >> 1);
        x = _mm_add_epi32(x, offset);
        x = _mm_sra_epi32(x, _mm_cvtsi32_si128(out_shift));
        x = _mm_max_epi32(x, clamp_lo_out);
        x = _mm_min_epi32(x, clamp_hi_out);
    }

    out[0] = x;
    out[1] = x;
    out[2] = x;
    out[3] = x;
    out[4] = x;
    out[5] = x;
    out[6] = x;
    out[7] = x;
}

static void iadst8x8_new_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi36 = _mm_set1_epi32(cospi[36]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospi52 = _mm_set1_epi32(cospi[52]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const __m128i kZero = _mm_setzero_si128();
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u[8], v[8], x;

    // stage 0
    // stage 1
    // stage 2

    u[0] = _mm_mullo_epi32(in[7], cospi4);
    x = _mm_mullo_epi32(in[0], cospi60);
    u[0] = _mm_add_epi32(u[0], x);
    u[0] = _mm_add_epi32(u[0], rnding);
    u[0] = _mm_srai_epi32(u[0], bit);

    u[1] = _mm_mullo_epi32(in[7], cospi60);
    x = _mm_mullo_epi32(in[0], cospi4);
    u[1] = _mm_sub_epi32(u[1], x);
    u[1] = _mm_add_epi32(u[1], rnding);
    u[1] = _mm_srai_epi32(u[1], bit);

    // (2)
    u[2] = _mm_mullo_epi32(in[5], cospi20);
    x = _mm_mullo_epi32(in[2], cospi44);
    u[2] = _mm_add_epi32(u[2], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_mullo_epi32(in[5], cospi44);
    x = _mm_mullo_epi32(in[2], cospi20);
    u[3] = _mm_sub_epi32(u[3], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    // (3)
    u[4] = _mm_mullo_epi32(in[3], cospi36);
    x = _mm_mullo_epi32(in[4], cospi28);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(in[3], cospi28);
    x = _mm_mullo_epi32(in[4], cospi36);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    // (4)
    u[6] = _mm_mullo_epi32(in[1], cospi52);
    x = _mm_mullo_epi32(in[6], cospi12);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(in[1], cospi12);
    x = _mm_mullo_epi32(in[6], cospi52);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 3
    addsub_sse4_1(u[0], u[4], &v[0], &v[4], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[1], u[5], &v[1], &v[5], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[2], u[6], &v[2], &v[6], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[3], u[7], &v[3], &v[7], &clamp_lo, &clamp_hi);

    // stage 4
    u[0] = v[0];
    u[1] = v[1];
    u[2] = v[2];
    u[3] = v[3];

    u[4] = _mm_mullo_epi32(v[4], cospi16);
    x = _mm_mullo_epi32(v[5], cospi48);
    u[4] = _mm_add_epi32(u[4], x);
    u[4] = _mm_add_epi32(u[4], rnding);
    u[4] = _mm_srai_epi32(u[4], bit);

    u[5] = _mm_mullo_epi32(v[4], cospi48);
    x = _mm_mullo_epi32(v[5], cospi16);
    u[5] = _mm_sub_epi32(u[5], x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    u[6] = _mm_mullo_epi32(v[6], cospim48);
    x = _mm_mullo_epi32(v[7], cospi16);
    u[6] = _mm_add_epi32(u[6], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_mullo_epi32(v[6], cospi16);
    x = _mm_mullo_epi32(v[7], cospim48);
    u[7] = _mm_sub_epi32(u[7], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 5
    addsub_sse4_1(u[0], u[2], &v[0], &v[2], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[1], u[3], &v[1], &v[3], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[4], u[6], &v[4], &v[6], &clamp_lo, &clamp_hi);
    addsub_sse4_1(u[5], u[7], &v[5], &v[7], &clamp_lo, &clamp_hi);

    // stage 6
    u[0] = v[0];
    u[1] = v[1];
    u[4] = v[4];
    u[5] = v[5];

    v[0] = _mm_mullo_epi32(v[2], cospi32);
    x = _mm_mullo_epi32(v[3], cospi32);
    u[2] = _mm_add_epi32(v[0], x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_sub_epi32(v[0], x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    v[0] = _mm_mullo_epi32(v[6], cospi32);
    x = _mm_mullo_epi32(v[7], cospi32);
    u[6] = _mm_add_epi32(v[0], x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_sub_epi32(v[0], x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 7
    if (do_cols) {
        out[0] = u[0];
        out[1] = _mm_sub_epi32(kZero, u[4]);
        out[2] = u[6];
        out[3] = _mm_sub_epi32(kZero, u[2]);
        out[4] = u[3];
        out[5] = _mm_sub_epi32(kZero, u[7]);
        out[6] = u[5];
        out[7] = _mm_sub_epi32(kZero, u[1]);
    }
    else {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(-(1 << (log_range_out - 1)));
        const __m128i clamp_hi_out = _mm_set1_epi32((1 << (log_range_out - 1)) - 1);

        neg_shift_sse4_1(u[0], u[4], out + 0, out + 1, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[6], u[2], out + 2, out + 3, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[3], u[7], out + 4, out + 5, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[5], u[1], out + 6, out + 7, &clamp_lo_out, &clamp_hi_out,
            out_shift);
    }
}

static void iadst8x8_low1_sse4_1(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const __m128i kZero = _mm_setzero_si128();
    __m128i u[8], x;

    // stage 0
    // stage 1
    // stage 2

    x = _mm_mullo_epi32(in[0], cospi60);
    u[0] = _mm_add_epi32(x, rnding);
    u[0] = _mm_srai_epi32(u[0], bit);

    x = _mm_mullo_epi32(in[0], cospi4);
    u[1] = _mm_sub_epi32(kZero, x);
    u[1] = _mm_add_epi32(u[1], rnding);
    u[1] = _mm_srai_epi32(u[1], bit);

    // stage 3
    // stage 4
    __m128i temp1, temp2;
    temp1 = _mm_mullo_epi32(u[0], cospi16);
    x = _mm_mullo_epi32(u[1], cospi48);
    temp1 = _mm_add_epi32(temp1, x);
    temp1 = _mm_add_epi32(temp1, rnding);
    temp1 = _mm_srai_epi32(temp1, bit);
    u[4] = temp1;

    temp2 = _mm_mullo_epi32(u[0], cospi48);
    x = _mm_mullo_epi32(u[1], cospi16);
    u[5] = _mm_sub_epi32(temp2, x);
    u[5] = _mm_add_epi32(u[5], rnding);
    u[5] = _mm_srai_epi32(u[5], bit);

    // stage 5
    // stage 6
    temp1 = _mm_mullo_epi32(u[0], cospi32);
    x = _mm_mullo_epi32(u[1], cospi32);
    u[2] = _mm_add_epi32(temp1, x);
    u[2] = _mm_add_epi32(u[2], rnding);
    u[2] = _mm_srai_epi32(u[2], bit);

    u[3] = _mm_sub_epi32(temp1, x);
    u[3] = _mm_add_epi32(u[3], rnding);
    u[3] = _mm_srai_epi32(u[3], bit);

    temp1 = _mm_mullo_epi32(u[4], cospi32);
    x = _mm_mullo_epi32(u[5], cospi32);
    u[6] = _mm_add_epi32(temp1, x);
    u[6] = _mm_add_epi32(u[6], rnding);
    u[6] = _mm_srai_epi32(u[6], bit);

    u[7] = _mm_sub_epi32(temp1, x);
    u[7] = _mm_add_epi32(u[7], rnding);
    u[7] = _mm_srai_epi32(u[7], bit);

    // stage 7
    if (do_cols) {
        out[0] = u[0];
        out[1] = _mm_sub_epi32(kZero, u[4]);
        out[2] = u[6];
        out[3] = _mm_sub_epi32(kZero, u[2]);
        out[4] = u[3];
        out[5] = _mm_sub_epi32(kZero, u[7]);
        out[6] = u[5];
        out[7] = _mm_sub_epi32(kZero, u[1]);
    }
    else {
        const int32_t log_range_out = AOMMAX(16, bd + 6);
        const __m128i clamp_lo_out = _mm_set1_epi32(-(1 << (log_range_out - 1)));
        const __m128i clamp_hi_out = _mm_set1_epi32((1 << (log_range_out - 1)) - 1);

        neg_shift_sse4_1(u[0], u[4], out + 0, out + 1, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[6], u[2], out + 2, out + 3, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[3], u[7], out + 4, out + 5, &clamp_lo_out, &clamp_hi_out,
            out_shift);
        neg_shift_sse4_1(u[5], u[1], out + 6, out + 7, &clamp_lo_out, &clamp_hi_out,
            out_shift);
    }
}

static void idct16x16_low1_sse4_1(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);

    {
        // stage 0
        // stage 1
        // stage 2
        // stage 3
        // stage 4
        in[0] = _mm_mullo_epi32(in[0], cospi32);
        in[0] = _mm_add_epi32(in[0], rnding);
        in[0] = _mm_srai_epi32(in[0], bit);

        // stage 5
        // stage 6
        // stage 7
        if (do_cols) {
            in[0] = _mm_max_epi32(in[0], clamp_lo);
            in[0] = _mm_min_epi32(in[0], clamp_hi);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
                -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
            const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
                (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));
            __m128i offset = _mm_set1_epi32((1 << out_shift) >> 1);
            in[0] = _mm_add_epi32(in[0], offset);
            in[0] = _mm_sra_epi32(in[0], _mm_cvtsi32_si128(out_shift));
            in[0] = _mm_max_epi32(in[0], clamp_lo_out);
            in[0] = _mm_min_epi32(in[0], clamp_hi_out);
        }

        out[0] = in[0];
        out[1] = in[0];
        out[2] = in[0];
        out[3] = in[0];
        out[4] = in[0];
        out[5] = in[0];
        out[6] = in[0];
        out[7] = in[0];
        out[8] = in[0];
        out[9] = in[0];
        out[10] = in[0];
        out[11] = in[0];
        out[12] = in[0];
        out[13] = in[0];
        out[14] = in[0];
        out[15] = in[0];
    }
}

static void idct16x16_low8_sse4_1(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospim36 = _mm_set1_epi32(-cospi[36]);
    const __m128i cospim52 = _mm_set1_epi32(-cospi[52]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u[16], x, y;

    {
        // stage 0
        // stage 1
        u[0] = in[0];
        u[2] = in[4];
        u[4] = in[2];
        u[6] = in[6];
        u[8] = in[1];
        u[10] = in[5];
        u[12] = in[3];
        u[14] = in[7];

        // stage 2
        u[15] = half_btf_0_sse4_1(&cospi4, &u[8], &rnding, bit);
        u[8] = half_btf_0_sse4_1(&cospi60, &u[8], &rnding, bit);

        u[9] = half_btf_0_sse4_1(&cospim36, &u[14], &rnding, bit);
        u[14] = half_btf_0_sse4_1(&cospi28, &u[14], &rnding, bit);

        u[13] = half_btf_0_sse4_1(&cospi20, &u[10], &rnding, bit);
        u[10] = half_btf_0_sse4_1(&cospi44, &u[10], &rnding, bit);

        u[11] = half_btf_0_sse4_1(&cospim52, &u[12], &rnding, bit);
        u[12] = half_btf_0_sse4_1(&cospi12, &u[12], &rnding, bit);

        // stage 3
        u[7] = half_btf_0_sse4_1(&cospi8, &u[4], &rnding, bit);
        u[4] = half_btf_0_sse4_1(&cospi56, &u[4], &rnding, bit);
        u[5] = half_btf_0_sse4_1(&cospim40, &u[6], &rnding, bit);
        u[6] = half_btf_0_sse4_1(&cospi24, &u[6], &rnding, bit);

        addsub_sse4_1(u[8], u[9], &u[8], &u[9], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[11], u[10], &u[11], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[12], u[13], &u[12], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[15], u[14], &u[15], &u[14], &clamp_lo, &clamp_hi);

        // stage 4
        x = _mm_mullo_epi32(u[0], cospi32);
        u[0] = _mm_add_epi32(x, rnding);
        u[0] = _mm_srai_epi32(u[0], bit);
        u[1] = u[0];

        u[3] = half_btf_0_sse4_1(&cospi16, &u[2], &rnding, bit);
        u[2] = half_btf_0_sse4_1(&cospi48, &u[2], &rnding, bit);

        addsub_sse4_1(u[4], u[5], &u[4], &u[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[7], u[6], &u[7], &u[6], &clamp_lo, &clamp_hi);

        x = half_btf_sse4_1(&cospim16, &u[9], &cospi48, &u[14], &rnding, bit);
        u[14] = half_btf_sse4_1(&cospi48, &u[9], &cospi16, &u[14], &rnding, bit);
        u[9] = x;
        y = half_btf_sse4_1(&cospim48, &u[10], &cospim16, &u[13], &rnding, bit);
        u[13] = half_btf_sse4_1(&cospim16, &u[10], &cospi48, &u[13], &rnding, bit);
        u[10] = y;

        // stage 5
        addsub_sse4_1(u[0], u[3], &u[0], &u[3], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[2], &u[1], &u[2], &clamp_lo, &clamp_hi);

        x = _mm_mullo_epi32(u[5], cospi32);
        y = _mm_mullo_epi32(u[6], cospi32);
        u[5] = _mm_sub_epi32(y, x);
        u[5] = _mm_add_epi32(u[5], rnding);
        u[5] = _mm_srai_epi32(u[5], bit);

        u[6] = _mm_add_epi32(y, x);
        u[6] = _mm_add_epi32(u[6], rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        addsub_sse4_1(u[8], u[11], &u[8], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[9], u[10], &u[9], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[15], u[12], &u[15], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[14], u[13], &u[14], &u[13], &clamp_lo, &clamp_hi);

        // stage 6
        addsub_sse4_1(u[0], u[7], &u[0], &u[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[6], &u[1], &u[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[2], u[5], &u[2], &u[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[3], u[4], &u[3], &u[4], &clamp_lo, &clamp_hi);

        x = _mm_mullo_epi32(u[10], cospi32);
        y = _mm_mullo_epi32(u[13], cospi32);
        u[10] = _mm_sub_epi32(y, x);
        u[10] = _mm_add_epi32(u[10], rnding);
        u[10] = _mm_srai_epi32(u[10], bit);

        u[13] = _mm_add_epi32(x, y);
        u[13] = _mm_add_epi32(u[13], rnding);
        u[13] = _mm_srai_epi32(u[13], bit);

        x = _mm_mullo_epi32(u[11], cospi32);
        y = _mm_mullo_epi32(u[12], cospi32);
        u[11] = _mm_sub_epi32(y, x);
        u[11] = _mm_add_epi32(u[11], rnding);
        u[11] = _mm_srai_epi32(u[11], bit);

        u[12] = _mm_add_epi32(x, y);
        u[12] = _mm_add_epi32(u[12], rnding);
        u[12] = _mm_srai_epi32(u[12], bit);
        // stage 7
        if (do_cols) {
            addsub_no_clamp_sse4_1(u[0], u[15], out + 0, out + 15);
            addsub_no_clamp_sse4_1(u[1], u[14], out + 1, out + 14);
            addsub_no_clamp_sse4_1(u[2], u[13], out + 2, out + 13);
            addsub_no_clamp_sse4_1(u[3], u[12], out + 3, out + 12);
            addsub_no_clamp_sse4_1(u[4], u[11], out + 4, out + 11);
            addsub_no_clamp_sse4_1(u[5], u[10], out + 5, out + 10);
            addsub_no_clamp_sse4_1(u[6], u[9], out + 6, out + 9);
            addsub_no_clamp_sse4_1(u[7], u[8], out + 7, out + 8);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
                -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
            const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
                (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));

            addsub_shift_sse4_1(u[0], u[15], out + 0, out + 15, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[1], u[14], out + 1, out + 14, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[2], u[13], out + 2, out + 13, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[3], u[12], out + 3, out + 12, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[4], u[11], out + 4, out + 11, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[5], u[10], out + 5, out + 10, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[6], u[9], out + 6, out + 9, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(u[7], u[8], out + 7, out + 8, &clamp_lo_out,
                &clamp_hi_out, out_shift);
        }
    }
}

static void idct16x16_new_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi60 = _mm_set1_epi32(cospi[60]);
    const __m128i cospim4 = _mm_set1_epi32(-cospi[4]);
    const __m128i cospi28 = _mm_set1_epi32(cospi[28]);
    const __m128i cospim36 = _mm_set1_epi32(-cospi[36]);
    const __m128i cospi44 = _mm_set1_epi32(cospi[44]);
    const __m128i cospi20 = _mm_set1_epi32(cospi[20]);
    const __m128i cospim20 = _mm_set1_epi32(-cospi[20]);
    const __m128i cospi12 = _mm_set1_epi32(cospi[12]);
    const __m128i cospim52 = _mm_set1_epi32(-cospi[52]);
    const __m128i cospi52 = _mm_set1_epi32(cospi[52]);
    const __m128i cospi36 = _mm_set1_epi32(cospi[36]);
    const __m128i cospi4 = _mm_set1_epi32(cospi[4]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospim8 = _mm_set1_epi32(-cospi[8]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim40 = _mm_set1_epi32(-cospi[40]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim16 = _mm_set1_epi32(-cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u[16], v[16], x, y;

    {
        // stage 0
        // stage 1
        u[0] = in[0];
        u[1] = in[8];
        u[2] = in[4];
        u[3] = in[12];
        u[4] = in[2];
        u[5] = in[10];
        u[6] = in[6];
        u[7] = in[14];
        u[8] = in[1];
        u[9] = in[9];
        u[10] = in[5];
        u[11] = in[13];
        u[12] = in[3];
        u[13] = in[11];
        u[14] = in[7];
        u[15] = in[15];

        // stage 2
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];
        v[4] = u[4];
        v[5] = u[5];
        v[6] = u[6];
        v[7] = u[7];

        v[8] = half_btf_sse4_1(&cospi60, &u[8], &cospim4, &u[15], &rnding, bit);
        v[9] = half_btf_sse4_1(&cospi28, &u[9], &cospim36, &u[14], &rnding, bit);
        v[10] = half_btf_sse4_1(&cospi44, &u[10], &cospim20, &u[13], &rnding, bit);
        v[11] = half_btf_sse4_1(&cospi12, &u[11], &cospim52, &u[12], &rnding, bit);
        v[12] = half_btf_sse4_1(&cospi52, &u[11], &cospi12, &u[12], &rnding, bit);
        v[13] = half_btf_sse4_1(&cospi20, &u[10], &cospi44, &u[13], &rnding, bit);
        v[14] = half_btf_sse4_1(&cospi36, &u[9], &cospi28, &u[14], &rnding, bit);
        v[15] = half_btf_sse4_1(&cospi4, &u[8], &cospi60, &u[15], &rnding, bit);

        // stage 3
        u[0] = v[0];
        u[1] = v[1];
        u[2] = v[2];
        u[3] = v[3];
        u[4] = half_btf_sse4_1(&cospi56, &v[4], &cospim8, &v[7], &rnding, bit);
        u[5] = half_btf_sse4_1(&cospi24, &v[5], &cospim40, &v[6], &rnding, bit);
        u[6] = half_btf_sse4_1(&cospi40, &v[5], &cospi24, &v[6], &rnding, bit);
        u[7] = half_btf_sse4_1(&cospi8, &v[4], &cospi56, &v[7], &rnding, bit);
        addsub_sse4_1(v[8], v[9], &u[8], &u[9], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[11], v[10], &u[11], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[12], v[13], &u[12], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[15], v[14], &u[15], &u[14], &clamp_lo, &clamp_hi);

        // stage 4
        x = _mm_mullo_epi32(u[0], cospi32);
        y = _mm_mullo_epi32(u[1], cospi32);
        v[0] = _mm_add_epi32(x, y);
        v[0] = _mm_add_epi32(v[0], rnding);
        v[0] = _mm_srai_epi32(v[0], bit);

        v[1] = _mm_sub_epi32(x, y);
        v[1] = _mm_add_epi32(v[1], rnding);
        v[1] = _mm_srai_epi32(v[1], bit);

        v[2] = half_btf_sse4_1(&cospi48, &u[2], &cospim16, &u[3], &rnding, bit);
        v[3] = half_btf_sse4_1(&cospi16, &u[2], &cospi48, &u[3], &rnding, bit);
        addsub_sse4_1(u[4], u[5], &v[4], &v[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[7], u[6], &v[7], &v[6], &clamp_lo, &clamp_hi);
        v[8] = u[8];
        v[9] = half_btf_sse4_1(&cospim16, &u[9], &cospi48, &u[14], &rnding, bit);
        v[10] = half_btf_sse4_1(&cospim48, &u[10], &cospim16, &u[13], &rnding, bit);
        v[11] = u[11];
        v[12] = u[12];
        v[13] = half_btf_sse4_1(&cospim16, &u[10], &cospi48, &u[13], &rnding, bit);
        v[14] = half_btf_sse4_1(&cospi48, &u[9], &cospi16, &u[14], &rnding, bit);
        v[15] = u[15];

        // stage 5
        addsub_sse4_1(v[0], v[3], &u[0], &u[3], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[1], v[2], &u[1], &u[2], &clamp_lo, &clamp_hi);
        u[4] = v[4];

        x = _mm_mullo_epi32(v[5], cospi32);
        y = _mm_mullo_epi32(v[6], cospi32);
        u[5] = _mm_sub_epi32(y, x);
        u[5] = _mm_add_epi32(u[5], rnding);
        u[5] = _mm_srai_epi32(u[5], bit);

        u[6] = _mm_add_epi32(y, x);
        u[6] = _mm_add_epi32(u[6], rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        u[7] = v[7];
        addsub_sse4_1(v[8], v[11], &u[8], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[9], v[10], &u[9], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[15], v[12], &u[15], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[14], v[13], &u[14], &u[13], &clamp_lo, &clamp_hi);

        // stage 6
        addsub_sse4_1(u[0], u[7], &v[0], &v[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[6], &v[1], &v[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[2], u[5], &v[2], &v[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[3], u[4], &v[3], &v[4], &clamp_lo, &clamp_hi);
        v[8] = u[8];
        v[9] = u[9];

        x = _mm_mullo_epi32(u[10], cospi32);
        y = _mm_mullo_epi32(u[13], cospi32);
        v[10] = _mm_sub_epi32(y, x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[13] = _mm_add_epi32(x, y);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        x = _mm_mullo_epi32(u[11], cospi32);
        y = _mm_mullo_epi32(u[12], cospi32);
        v[11] = _mm_sub_epi32(y, x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_add_epi32(x, y);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[14] = u[14];
        v[15] = u[15];

        // stage 7
        if (do_cols) {
            addsub_no_clamp_sse4_1(v[0], v[15], out + 0, out + 15);
            addsub_no_clamp_sse4_1(v[1], v[14], out + 1, out + 14);
            addsub_no_clamp_sse4_1(v[2], v[13], out + 2, out + 13);
            addsub_no_clamp_sse4_1(v[3], v[12], out + 3, out + 12);
            addsub_no_clamp_sse4_1(v[4], v[11], out + 4, out + 11);
            addsub_no_clamp_sse4_1(v[5], v[10], out + 5, out + 10);
            addsub_no_clamp_sse4_1(v[6], v[9], out + 6, out + 9);
            addsub_no_clamp_sse4_1(v[7], v[8], out + 7, out + 8);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(AOMMAX(
                -(1 << (log_range_out - 1)), -(1 << (log_range - 1 - out_shift))));
            const __m128i clamp_hi_out = _mm_set1_epi32(AOMMIN(
                (1 << (log_range_out - 1)) - 1, (1 << (log_range - 1 - out_shift))));

            addsub_shift_sse4_1(v[0], v[15], out + 0, out + 15, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[1], v[14], out + 1, out + 14, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[2], v[13], out + 2, out + 13, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[3], v[12], out + 3, out + 12, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[4], v[11], out + 4, out + 11, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[5], v[10], out + 5, out + 10, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[6], v[9], out + 6, out + 9, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            addsub_shift_sse4_1(v[7], v[8], out + 7, out + 8, &clamp_lo_out,
                &clamp_hi_out, out_shift);
        }
    }
}

static void iadst16x16_new_sse4_1(__m128i *in, __m128i *out, int32_t bit, int32_t do_cols,
    int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi2 = _mm_set1_epi32(cospi[2]);
    const __m128i cospi62 = _mm_set1_epi32(cospi[62]);
    const __m128i cospi10 = _mm_set1_epi32(cospi[10]);
    const __m128i cospi54 = _mm_set1_epi32(cospi[54]);
    const __m128i cospi18 = _mm_set1_epi32(cospi[18]);
    const __m128i cospi46 = _mm_set1_epi32(cospi[46]);
    const __m128i cospi26 = _mm_set1_epi32(cospi[26]);
    const __m128i cospi38 = _mm_set1_epi32(cospi[38]);
    const __m128i cospi34 = _mm_set1_epi32(cospi[34]);
    const __m128i cospi30 = _mm_set1_epi32(cospi[30]);
    const __m128i cospi42 = _mm_set1_epi32(cospi[42]);
    const __m128i cospi22 = _mm_set1_epi32(cospi[22]);
    const __m128i cospi50 = _mm_set1_epi32(cospi[50]);
    const __m128i cospi14 = _mm_set1_epi32(cospi[14]);
    const __m128i cospi58 = _mm_set1_epi32(cospi[58]);
    const __m128i cospi6 = _mm_set1_epi32(cospi[6]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim56 = _mm_set1_epi32(-cospi[56]);
    const __m128i cospim24 = _mm_set1_epi32(-cospi[24]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u[16], v[16], x, y;

    // Calculate the column 0, 1, 2, 3
    {
        // stage 0
        // stage 1
        // stage 2
        v[0] = _mm_mullo_epi32(in[15], cospi2);
        x = _mm_mullo_epi32(in[0], cospi62);
        v[0] = _mm_add_epi32(v[0], x);
        v[0] = _mm_add_epi32(v[0], rnding);
        v[0] = _mm_srai_epi32(v[0], bit);

        v[1] = _mm_mullo_epi32(in[15], cospi62);
        x = _mm_mullo_epi32(in[0], cospi2);
        v[1] = _mm_sub_epi32(v[1], x);
        v[1] = _mm_add_epi32(v[1], rnding);
        v[1] = _mm_srai_epi32(v[1], bit);

        v[2] = _mm_mullo_epi32(in[13], cospi10);
        x = _mm_mullo_epi32(in[2], cospi54);
        v[2] = _mm_add_epi32(v[2], x);
        v[2] = _mm_add_epi32(v[2], rnding);
        v[2] = _mm_srai_epi32(v[2], bit);

        v[3] = _mm_mullo_epi32(in[13], cospi54);
        x = _mm_mullo_epi32(in[2], cospi10);
        v[3] = _mm_sub_epi32(v[3], x);
        v[3] = _mm_add_epi32(v[3], rnding);
        v[3] = _mm_srai_epi32(v[3], bit);

        v[4] = _mm_mullo_epi32(in[11], cospi18);
        x = _mm_mullo_epi32(in[4], cospi46);
        v[4] = _mm_add_epi32(v[4], x);
        v[4] = _mm_add_epi32(v[4], rnding);
        v[4] = _mm_srai_epi32(v[4], bit);

        v[5] = _mm_mullo_epi32(in[11], cospi46);
        x = _mm_mullo_epi32(in[4], cospi18);
        v[5] = _mm_sub_epi32(v[5], x);
        v[5] = _mm_add_epi32(v[5], rnding);
        v[5] = _mm_srai_epi32(v[5], bit);

        v[6] = _mm_mullo_epi32(in[9], cospi26);
        x = _mm_mullo_epi32(in[6], cospi38);
        v[6] = _mm_add_epi32(v[6], x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_mullo_epi32(in[9], cospi38);
        x = _mm_mullo_epi32(in[6], cospi26);
        v[7] = _mm_sub_epi32(v[7], x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = _mm_mullo_epi32(in[7], cospi34);
        x = _mm_mullo_epi32(in[8], cospi30);
        v[8] = _mm_add_epi32(v[8], x);
        v[8] = _mm_add_epi32(v[8], rnding);
        v[8] = _mm_srai_epi32(v[8], bit);

        v[9] = _mm_mullo_epi32(in[7], cospi30);
        x = _mm_mullo_epi32(in[8], cospi34);
        v[9] = _mm_sub_epi32(v[9], x);
        v[9] = _mm_add_epi32(v[9], rnding);
        v[9] = _mm_srai_epi32(v[9], bit);

        v[10] = _mm_mullo_epi32(in[5], cospi42);
        x = _mm_mullo_epi32(in[10], cospi22);
        v[10] = _mm_add_epi32(v[10], x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_mullo_epi32(in[5], cospi22);
        x = _mm_mullo_epi32(in[10], cospi42);
        v[11] = _mm_sub_epi32(v[11], x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_mullo_epi32(in[3], cospi50);
        x = _mm_mullo_epi32(in[12], cospi14);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(in[3], cospi14);
        x = _mm_mullo_epi32(in[12], cospi50);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(in[1], cospi58);
        x = _mm_mullo_epi32(in[14], cospi6);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(in[1], cospi6);
        x = _mm_mullo_epi32(in[14], cospi58);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 3
        addsub_sse4_1(v[0], v[8], &u[0], &u[8], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[1], v[9], &u[1], &u[9], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[2], v[10], &u[2], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[3], v[11], &u[3], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[4], v[12], &u[4], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[5], v[13], &u[5], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[6], v[14], &u[6], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[7], v[15], &u[7], &u[15], &clamp_lo, &clamp_hi);

        // stage 4
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];
        v[4] = u[4];
        v[5] = u[5];
        v[6] = u[6];
        v[7] = u[7];

        v[8] = _mm_mullo_epi32(u[8], cospi8);
        x = _mm_mullo_epi32(u[9], cospi56);
        v[8] = _mm_add_epi32(v[8], x);
        v[8] = _mm_add_epi32(v[8], rnding);
        v[8] = _mm_srai_epi32(v[8], bit);

        v[9] = _mm_mullo_epi32(u[8], cospi56);
        x = _mm_mullo_epi32(u[9], cospi8);
        v[9] = _mm_sub_epi32(v[9], x);
        v[9] = _mm_add_epi32(v[9], rnding);
        v[9] = _mm_srai_epi32(v[9], bit);

        v[10] = _mm_mullo_epi32(u[10], cospi40);
        x = _mm_mullo_epi32(u[11], cospi24);
        v[10] = _mm_add_epi32(v[10], x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_mullo_epi32(u[10], cospi24);
        x = _mm_mullo_epi32(u[11], cospi40);
        v[11] = _mm_sub_epi32(v[11], x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = _mm_mullo_epi32(u[12], cospim56);
        x = _mm_mullo_epi32(u[13], cospi8);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(u[12], cospi8);
        x = _mm_mullo_epi32(u[13], cospim56);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(u[14], cospim24);
        x = _mm_mullo_epi32(u[15], cospi40);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(u[14], cospi40);
        x = _mm_mullo_epi32(u[15], cospim24);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 5
        addsub_sse4_1(v[0], v[4], &u[0], &u[4], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[1], v[5], &u[1], &u[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[2], v[6], &u[2], &u[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[3], v[7], &u[3], &u[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[8], v[12], &u[8], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[9], v[13], &u[9], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[10], v[14], &u[10], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[11], v[15], &u[11], &u[15], &clamp_lo, &clamp_hi);

        // stage 6
        v[0] = u[0];
        v[1] = u[1];
        v[2] = u[2];
        v[3] = u[3];

        v[4] = _mm_mullo_epi32(u[4], cospi16);
        x = _mm_mullo_epi32(u[5], cospi48);
        v[4] = _mm_add_epi32(v[4], x);
        v[4] = _mm_add_epi32(v[4], rnding);
        v[4] = _mm_srai_epi32(v[4], bit);

        v[5] = _mm_mullo_epi32(u[4], cospi48);
        x = _mm_mullo_epi32(u[5], cospi16);
        v[5] = _mm_sub_epi32(v[5], x);
        v[5] = _mm_add_epi32(v[5], rnding);
        v[5] = _mm_srai_epi32(v[5], bit);

        v[6] = _mm_mullo_epi32(u[6], cospim48);
        x = _mm_mullo_epi32(u[7], cospi16);
        v[6] = _mm_add_epi32(v[6], x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_mullo_epi32(u[6], cospi16);
        x = _mm_mullo_epi32(u[7], cospim48);
        v[7] = _mm_sub_epi32(v[7], x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = u[8];
        v[9] = u[9];
        v[10] = u[10];
        v[11] = u[11];

        v[12] = _mm_mullo_epi32(u[12], cospi16);
        x = _mm_mullo_epi32(u[13], cospi48);
        v[12] = _mm_add_epi32(v[12], x);
        v[12] = _mm_add_epi32(v[12], rnding);
        v[12] = _mm_srai_epi32(v[12], bit);

        v[13] = _mm_mullo_epi32(u[12], cospi48);
        x = _mm_mullo_epi32(u[13], cospi16);
        v[13] = _mm_sub_epi32(v[13], x);
        v[13] = _mm_add_epi32(v[13], rnding);
        v[13] = _mm_srai_epi32(v[13], bit);

        v[14] = _mm_mullo_epi32(u[14], cospim48);
        x = _mm_mullo_epi32(u[15], cospi16);
        v[14] = _mm_add_epi32(v[14], x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_mullo_epi32(u[14], cospi16);
        x = _mm_mullo_epi32(u[15], cospim48);
        v[15] = _mm_sub_epi32(v[15], x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 7
        addsub_sse4_1(v[0], v[2], &u[0], &u[2], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[1], v[3], &u[1], &u[3], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[4], v[6], &u[4], &u[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[5], v[7], &u[5], &u[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[8], v[10], &u[8], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[9], v[11], &u[9], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[12], v[14], &u[12], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(v[13], v[15], &u[13], &u[15], &clamp_lo, &clamp_hi);

        // stage 8
        v[0] = u[0];
        v[1] = u[1];

        y = _mm_mullo_epi32(u[2], cospi32);
        x = _mm_mullo_epi32(u[3], cospi32);
        v[2] = _mm_add_epi32(y, x);
        v[2] = _mm_add_epi32(v[2], rnding);
        v[2] = _mm_srai_epi32(v[2], bit);

        v[3] = _mm_sub_epi32(y, x);
        v[3] = _mm_add_epi32(v[3], rnding);
        v[3] = _mm_srai_epi32(v[3], bit);

        v[4] = u[4];
        v[5] = u[5];

        y = _mm_mullo_epi32(u[6], cospi32);
        x = _mm_mullo_epi32(u[7], cospi32);
        v[6] = _mm_add_epi32(y, x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_sub_epi32(y, x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        v[8] = u[8];
        v[9] = u[9];

        y = _mm_mullo_epi32(u[10], cospi32);
        x = _mm_mullo_epi32(u[11], cospi32);
        v[10] = _mm_add_epi32(y, x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_sub_epi32(y, x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        v[12] = u[12];
        v[13] = u[13];

        y = _mm_mullo_epi32(u[14], cospi32);
        x = _mm_mullo_epi32(u[15], cospi32);
        v[14] = _mm_add_epi32(y, x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_sub_epi32(y, x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 9
        if (do_cols) {
            out[0] = v[0];
            out[1] = _mm_sub_epi32(_mm_setzero_si128(), v[8]);
            out[2] = v[12];
            out[3] = _mm_sub_epi32(_mm_setzero_si128(), v[4]);
            out[4] = v[6];
            out[5] = _mm_sub_epi32(_mm_setzero_si128(), v[14]);
            out[6] = v[10];
            out[7] = _mm_sub_epi32(_mm_setzero_si128(), v[2]);
            out[8] = v[3];
            out[9] = _mm_sub_epi32(_mm_setzero_si128(), v[11]);
            out[10] = v[15];
            out[11] = _mm_sub_epi32(_mm_setzero_si128(), v[7]);
            out[12] = v[5];
            out[13] = _mm_sub_epi32(_mm_setzero_si128(), v[13]);
            out[14] = v[9];
            out[15] = _mm_sub_epi32(_mm_setzero_si128(), v[1]);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(-(1 << (log_range_out - 1)));
            const __m128i clamp_hi_out =
                _mm_set1_epi32((1 << (log_range_out - 1)) - 1);

            neg_shift_sse4_1(v[0], v[8], out + 0, out + 1, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[12], v[4], out + 2, out + 3, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[6], v[14], out + 4, out + 5, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[10], v[2], out + 6, out + 7, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[3], v[11], out + 8, out + 9, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[15], v[7], out + 10, out + 11, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[5], v[13], out + 12, out + 13, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[9], v[1], out + 14, out + 15, &clamp_lo_out,
                &clamp_hi_out, out_shift);
        }
    }
}

static void iadst16x16_low1_sse4_1(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi2 = _mm_set1_epi32(cospi[2]);
    const __m128i cospi62 = _mm_set1_epi32(cospi[62]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const __m128i zero = _mm_setzero_si128();
    __m128i v[16], x, y, temp1, temp2;

    // Calculate the column 0, 1, 2, 3
    {
        // stage 0
        // stage 1
        // stage 2
        x = _mm_mullo_epi32(in[0], cospi62);
        v[0] = _mm_add_epi32(x, rnding);
        v[0] = _mm_srai_epi32(v[0], bit);

        x = _mm_mullo_epi32(in[0], cospi2);
        v[1] = _mm_sub_epi32(zero, x);
        v[1] = _mm_add_epi32(v[1], rnding);
        v[1] = _mm_srai_epi32(v[1], bit);

        // stage 3
        v[8] = v[0];
        v[9] = v[1];

        // stage 4
        temp1 = _mm_mullo_epi32(v[8], cospi8);
        x = _mm_mullo_epi32(v[9], cospi56);
        temp1 = _mm_add_epi32(temp1, x);
        temp1 = _mm_add_epi32(temp1, rnding);
        temp1 = _mm_srai_epi32(temp1, bit);

        temp2 = _mm_mullo_epi32(v[8], cospi56);
        x = _mm_mullo_epi32(v[9], cospi8);
        temp2 = _mm_sub_epi32(temp2, x);
        temp2 = _mm_add_epi32(temp2, rnding);
        temp2 = _mm_srai_epi32(temp2, bit);
        v[8] = temp1;
        v[9] = temp2;

        // stage 5
        v[4] = v[0];
        v[5] = v[1];
        v[12] = v[8];
        v[13] = v[9];

        // stage 6
        temp1 = _mm_mullo_epi32(v[4], cospi16);
        x = _mm_mullo_epi32(v[5], cospi48);
        temp1 = _mm_add_epi32(temp1, x);
        temp1 = _mm_add_epi32(temp1, rnding);
        temp1 = _mm_srai_epi32(temp1, bit);

        temp2 = _mm_mullo_epi32(v[4], cospi48);
        x = _mm_mullo_epi32(v[5], cospi16);
        temp2 = _mm_sub_epi32(temp2, x);
        temp2 = _mm_add_epi32(temp2, rnding);
        temp2 = _mm_srai_epi32(temp2, bit);
        v[4] = temp1;
        v[5] = temp2;

        temp1 = _mm_mullo_epi32(v[12], cospi16);
        x = _mm_mullo_epi32(v[13], cospi48);
        temp1 = _mm_add_epi32(temp1, x);
        temp1 = _mm_add_epi32(temp1, rnding);
        temp1 = _mm_srai_epi32(temp1, bit);

        temp2 = _mm_mullo_epi32(v[12], cospi48);
        x = _mm_mullo_epi32(v[13], cospi16);
        temp2 = _mm_sub_epi32(temp2, x);
        temp2 = _mm_add_epi32(temp2, rnding);
        temp2 = _mm_srai_epi32(temp2, bit);
        v[12] = temp1;
        v[13] = temp2;

        // stage 7
        v[2] = v[0];
        v[3] = v[1];
        v[6] = v[4];
        v[7] = v[5];
        v[10] = v[8];
        v[11] = v[9];
        v[14] = v[12];
        v[15] = v[13];

        // stage 8
        y = _mm_mullo_epi32(v[2], cospi32);
        x = _mm_mullo_epi32(v[3], cospi32);
        v[2] = _mm_add_epi32(y, x);
        v[2] = _mm_add_epi32(v[2], rnding);
        v[2] = _mm_srai_epi32(v[2], bit);

        v[3] = _mm_sub_epi32(y, x);
        v[3] = _mm_add_epi32(v[3], rnding);
        v[3] = _mm_srai_epi32(v[3], bit);

        y = _mm_mullo_epi32(v[6], cospi32);
        x = _mm_mullo_epi32(v[7], cospi32);
        v[6] = _mm_add_epi32(y, x);
        v[6] = _mm_add_epi32(v[6], rnding);
        v[6] = _mm_srai_epi32(v[6], bit);

        v[7] = _mm_sub_epi32(y, x);
        v[7] = _mm_add_epi32(v[7], rnding);
        v[7] = _mm_srai_epi32(v[7], bit);

        y = _mm_mullo_epi32(v[10], cospi32);
        x = _mm_mullo_epi32(v[11], cospi32);
        v[10] = _mm_add_epi32(y, x);
        v[10] = _mm_add_epi32(v[10], rnding);
        v[10] = _mm_srai_epi32(v[10], bit);

        v[11] = _mm_sub_epi32(y, x);
        v[11] = _mm_add_epi32(v[11], rnding);
        v[11] = _mm_srai_epi32(v[11], bit);

        y = _mm_mullo_epi32(v[14], cospi32);
        x = _mm_mullo_epi32(v[15], cospi32);
        v[14] = _mm_add_epi32(y, x);
        v[14] = _mm_add_epi32(v[14], rnding);
        v[14] = _mm_srai_epi32(v[14], bit);

        v[15] = _mm_sub_epi32(y, x);
        v[15] = _mm_add_epi32(v[15], rnding);
        v[15] = _mm_srai_epi32(v[15], bit);

        // stage 9
        if (do_cols) {
            out[0] = v[0];
            out[1] = _mm_sub_epi32(_mm_setzero_si128(), v[8]);
            out[2] = v[12];
            out[3] = _mm_sub_epi32(_mm_setzero_si128(), v[4]);
            out[4] = v[6];
            out[5] = _mm_sub_epi32(_mm_setzero_si128(), v[14]);
            out[6] = v[10];
            out[7] = _mm_sub_epi32(_mm_setzero_si128(), v[2]);
            out[8] = v[3];
            out[9] = _mm_sub_epi32(_mm_setzero_si128(), v[11]);
            out[10] = v[15];
            out[11] = _mm_sub_epi32(_mm_setzero_si128(), v[7]);
            out[12] = v[5];
            out[13] = _mm_sub_epi32(_mm_setzero_si128(), v[13]);
            out[14] = v[9];
            out[15] = _mm_sub_epi32(_mm_setzero_si128(), v[1]);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(-(1 << (log_range_out - 1)));
            const __m128i clamp_hi_out =
                _mm_set1_epi32((1 << (log_range_out - 1)) - 1);

            neg_shift_sse4_1(v[0], v[8], out + 0, out + 1, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[12], v[4], out + 2, out + 3, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[6], v[14], out + 4, out + 5, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[10], v[2], out + 6, out + 7, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[3], v[11], out + 8, out + 9, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[15], v[7], out + 10, out + 11, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[5], v[13], out + 12, out + 13, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(v[9], v[1], out + 14, out + 15, &clamp_lo_out,
                &clamp_hi_out, out_shift);
        }
    }
}

static void iadst16x16_low8_sse4_1(__m128i *in, __m128i *out, int32_t bit,
    int32_t do_cols, int32_t bd, int32_t out_shift) {
    const int32_t *cospi = cospi_arr(bit);
    const __m128i cospi2 = _mm_set1_epi32(cospi[2]);
    const __m128i cospi62 = _mm_set1_epi32(cospi[62]);
    const __m128i cospi10 = _mm_set1_epi32(cospi[10]);
    const __m128i cospi54 = _mm_set1_epi32(cospi[54]);
    const __m128i cospi18 = _mm_set1_epi32(cospi[18]);
    const __m128i cospi46 = _mm_set1_epi32(cospi[46]);
    const __m128i cospi26 = _mm_set1_epi32(cospi[26]);
    const __m128i cospi38 = _mm_set1_epi32(cospi[38]);
    const __m128i cospi34 = _mm_set1_epi32(cospi[34]);
    const __m128i cospi30 = _mm_set1_epi32(cospi[30]);
    const __m128i cospi42 = _mm_set1_epi32(cospi[42]);
    const __m128i cospi22 = _mm_set1_epi32(cospi[22]);
    const __m128i cospi50 = _mm_set1_epi32(cospi[50]);
    const __m128i cospi14 = _mm_set1_epi32(cospi[14]);
    const __m128i cospi58 = _mm_set1_epi32(cospi[58]);
    const __m128i cospi6 = _mm_set1_epi32(cospi[6]);
    const __m128i cospi8 = _mm_set1_epi32(cospi[8]);
    const __m128i cospi56 = _mm_set1_epi32(cospi[56]);
    const __m128i cospi40 = _mm_set1_epi32(cospi[40]);
    const __m128i cospi24 = _mm_set1_epi32(cospi[24]);
    const __m128i cospim56 = _mm_set1_epi32(-cospi[56]);
    const __m128i cospim24 = _mm_set1_epi32(-cospi[24]);
    const __m128i cospi48 = _mm_set1_epi32(cospi[48]);
    const __m128i cospi16 = _mm_set1_epi32(cospi[16]);
    const __m128i cospim48 = _mm_set1_epi32(-cospi[48]);
    const __m128i cospi32 = _mm_set1_epi32(cospi[32]);
    const __m128i rnding = _mm_set1_epi32(1 << (bit - 1));
    const int32_t log_range = AOMMAX(16, bd + (do_cols ? 6 : 8));
    const __m128i clamp_lo = _mm_set1_epi32(-(1 << (log_range - 1)));
    const __m128i clamp_hi = _mm_set1_epi32((1 << (log_range - 1)) - 1);
    __m128i u[16], x, y;

    // Calculate the column 0, 1, 2, 3
    {
        // stage 0
        // stage 1
        // stage 2
        __m128i zero = _mm_setzero_si128();
        x = _mm_mullo_epi32(in[0], cospi62);
        u[0] = _mm_add_epi32(x, rnding);
        u[0] = _mm_srai_epi32(u[0], bit);

        x = _mm_mullo_epi32(in[0], cospi2);
        u[1] = _mm_sub_epi32(zero, x);
        u[1] = _mm_add_epi32(u[1], rnding);
        u[1] = _mm_srai_epi32(u[1], bit);

        x = _mm_mullo_epi32(in[2], cospi54);
        u[2] = _mm_add_epi32(x, rnding);
        u[2] = _mm_srai_epi32(u[2], bit);

        x = _mm_mullo_epi32(in[2], cospi10);
        u[3] = _mm_sub_epi32(zero, x);
        u[3] = _mm_add_epi32(u[3], rnding);
        u[3] = _mm_srai_epi32(u[3], bit);

        x = _mm_mullo_epi32(in[4], cospi46);
        u[4] = _mm_add_epi32(x, rnding);
        u[4] = _mm_srai_epi32(u[4], bit);

        x = _mm_mullo_epi32(in[4], cospi18);
        u[5] = _mm_sub_epi32(zero, x);
        u[5] = _mm_add_epi32(u[5], rnding);
        u[5] = _mm_srai_epi32(u[5], bit);

        x = _mm_mullo_epi32(in[6], cospi38);
        u[6] = _mm_add_epi32(x, rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        x = _mm_mullo_epi32(in[6], cospi26);
        u[7] = _mm_sub_epi32(zero, x);
        u[7] = _mm_add_epi32(u[7], rnding);
        u[7] = _mm_srai_epi32(u[7], bit);

        u[8] = _mm_mullo_epi32(in[7], cospi34);
        u[8] = _mm_add_epi32(u[8], rnding);
        u[8] = _mm_srai_epi32(u[8], bit);

        u[9] = _mm_mullo_epi32(in[7], cospi30);
        u[9] = _mm_add_epi32(u[9], rnding);
        u[9] = _mm_srai_epi32(u[9], bit);

        u[10] = _mm_mullo_epi32(in[5], cospi42);
        u[10] = _mm_add_epi32(u[10], rnding);
        u[10] = _mm_srai_epi32(u[10], bit);

        u[11] = _mm_mullo_epi32(in[5], cospi22);
        u[11] = _mm_add_epi32(u[11], rnding);
        u[11] = _mm_srai_epi32(u[11], bit);

        u[12] = _mm_mullo_epi32(in[3], cospi50);
        u[12] = _mm_add_epi32(u[12], rnding);
        u[12] = _mm_srai_epi32(u[12], bit);

        u[13] = _mm_mullo_epi32(in[3], cospi14);
        u[13] = _mm_add_epi32(u[13], rnding);
        u[13] = _mm_srai_epi32(u[13], bit);

        u[14] = _mm_mullo_epi32(in[1], cospi58);
        u[14] = _mm_add_epi32(u[14], rnding);
        u[14] = _mm_srai_epi32(u[14], bit);

        u[15] = _mm_mullo_epi32(in[1], cospi6);
        u[15] = _mm_add_epi32(u[15], rnding);
        u[15] = _mm_srai_epi32(u[15], bit);

        // stage 3
        addsub_sse4_1(u[0], u[8], &u[0], &u[8], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[9], &u[1], &u[9], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[2], u[10], &u[2], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[3], u[11], &u[3], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[4], u[12], &u[4], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[5], u[13], &u[5], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[6], u[14], &u[6], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[7], u[15], &u[7], &u[15], &clamp_lo, &clamp_hi);

        // stage 4
        y = _mm_mullo_epi32(u[8], cospi56);
        x = _mm_mullo_epi32(u[9], cospi56);
        u[8] = _mm_mullo_epi32(u[8], cospi8);
        u[8] = _mm_add_epi32(u[8], x);
        u[8] = _mm_add_epi32(u[8], rnding);
        u[8] = _mm_srai_epi32(u[8], bit);

        x = _mm_mullo_epi32(u[9], cospi8);
        u[9] = _mm_sub_epi32(y, x);
        u[9] = _mm_add_epi32(u[9], rnding);
        u[9] = _mm_srai_epi32(u[9], bit);

        x = _mm_mullo_epi32(u[11], cospi24);
        y = _mm_mullo_epi32(u[10], cospi24);
        u[10] = _mm_mullo_epi32(u[10], cospi40);
        u[10] = _mm_add_epi32(u[10], x);
        u[10] = _mm_add_epi32(u[10], rnding);
        u[10] = _mm_srai_epi32(u[10], bit);

        x = _mm_mullo_epi32(u[11], cospi40);
        u[11] = _mm_sub_epi32(y, x);
        u[11] = _mm_add_epi32(u[11], rnding);
        u[11] = _mm_srai_epi32(u[11], bit);

        x = _mm_mullo_epi32(u[13], cospi8);
        y = _mm_mullo_epi32(u[12], cospi8);
        u[12] = _mm_mullo_epi32(u[12], cospim56);
        u[12] = _mm_add_epi32(u[12], x);
        u[12] = _mm_add_epi32(u[12], rnding);
        u[12] = _mm_srai_epi32(u[12], bit);

        x = _mm_mullo_epi32(u[13], cospim56);
        u[13] = _mm_sub_epi32(y, x);
        u[13] = _mm_add_epi32(u[13], rnding);
        u[13] = _mm_srai_epi32(u[13], bit);

        x = _mm_mullo_epi32(u[15], cospi40);
        y = _mm_mullo_epi32(u[14], cospi40);
        u[14] = _mm_mullo_epi32(u[14], cospim24);
        u[14] = _mm_add_epi32(u[14], x);
        u[14] = _mm_add_epi32(u[14], rnding);
        u[14] = _mm_srai_epi32(u[14], bit);

        x = _mm_mullo_epi32(u[15], cospim24);
        u[15] = _mm_sub_epi32(y, x);
        u[15] = _mm_add_epi32(u[15], rnding);
        u[15] = _mm_srai_epi32(u[15], bit);

        // stage 5
        addsub_sse4_1(u[0], u[4], &u[0], &u[4], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[5], &u[1], &u[5], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[2], u[6], &u[2], &u[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[3], u[7], &u[3], &u[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[8], u[12], &u[8], &u[12], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[9], u[13], &u[9], &u[13], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[10], u[14], &u[10], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[11], u[15], &u[11], &u[15], &clamp_lo, &clamp_hi);

        // stage 6
        x = _mm_mullo_epi32(u[5], cospi48);
        y = _mm_mullo_epi32(u[4], cospi48);
        u[4] = _mm_mullo_epi32(u[4], cospi16);
        u[4] = _mm_add_epi32(u[4], x);
        u[4] = _mm_add_epi32(u[4], rnding);
        u[4] = _mm_srai_epi32(u[4], bit);

        x = _mm_mullo_epi32(u[5], cospi16);
        u[5] = _mm_sub_epi32(y, x);
        u[5] = _mm_add_epi32(u[5], rnding);
        u[5] = _mm_srai_epi32(u[5], bit);

        x = _mm_mullo_epi32(u[7], cospi16);
        y = _mm_mullo_epi32(u[6], cospi16);
        u[6] = _mm_mullo_epi32(u[6], cospim48);
        u[6] = _mm_add_epi32(u[6], x);
        u[6] = _mm_add_epi32(u[6], rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        x = _mm_mullo_epi32(u[7], cospim48);
        u[7] = _mm_sub_epi32(y, x);
        u[7] = _mm_add_epi32(u[7], rnding);
        u[7] = _mm_srai_epi32(u[7], bit);

        x = _mm_mullo_epi32(u[13], cospi48);
        y = _mm_mullo_epi32(u[12], cospi48);
        u[12] = _mm_mullo_epi32(u[12], cospi16);
        u[12] = _mm_add_epi32(u[12], x);
        u[12] = _mm_add_epi32(u[12], rnding);
        u[12] = _mm_srai_epi32(u[12], bit);

        x = _mm_mullo_epi32(u[13], cospi16);
        u[13] = _mm_sub_epi32(y, x);
        u[13] = _mm_add_epi32(u[13], rnding);
        u[13] = _mm_srai_epi32(u[13], bit);

        x = _mm_mullo_epi32(u[15], cospi16);
        y = _mm_mullo_epi32(u[14], cospi16);
        u[14] = _mm_mullo_epi32(u[14], cospim48);
        u[14] = _mm_add_epi32(u[14], x);
        u[14] = _mm_add_epi32(u[14], rnding);
        u[14] = _mm_srai_epi32(u[14], bit);

        x = _mm_mullo_epi32(u[15], cospim48);
        u[15] = _mm_sub_epi32(y, x);
        u[15] = _mm_add_epi32(u[15], rnding);
        u[15] = _mm_srai_epi32(u[15], bit);

        // stage 7
        addsub_sse4_1(u[0], u[2], &u[0], &u[2], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[1], u[3], &u[1], &u[3], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[4], u[6], &u[4], &u[6], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[5], u[7], &u[5], &u[7], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[8], u[10], &u[8], &u[10], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[9], u[11], &u[9], &u[11], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[12], u[14], &u[12], &u[14], &clamp_lo, &clamp_hi);
        addsub_sse4_1(u[13], u[15], &u[13], &u[15], &clamp_lo, &clamp_hi);

        // stage 8
        y = _mm_mullo_epi32(u[2], cospi32);
        x = _mm_mullo_epi32(u[3], cospi32);
        u[2] = _mm_add_epi32(y, x);
        u[2] = _mm_add_epi32(u[2], rnding);
        u[2] = _mm_srai_epi32(u[2], bit);

        u[3] = _mm_sub_epi32(y, x);
        u[3] = _mm_add_epi32(u[3], rnding);
        u[3] = _mm_srai_epi32(u[3], bit);
        y = _mm_mullo_epi32(u[6], cospi32);
        x = _mm_mullo_epi32(u[7], cospi32);
        u[6] = _mm_add_epi32(y, x);
        u[6] = _mm_add_epi32(u[6], rnding);
        u[6] = _mm_srai_epi32(u[6], bit);

        u[7] = _mm_sub_epi32(y, x);
        u[7] = _mm_add_epi32(u[7], rnding);
        u[7] = _mm_srai_epi32(u[7], bit);

        y = _mm_mullo_epi32(u[10], cospi32);
        x = _mm_mullo_epi32(u[11], cospi32);
        u[10] = _mm_add_epi32(y, x);
        u[10] = _mm_add_epi32(u[10], rnding);
        u[10] = _mm_srai_epi32(u[10], bit);

        u[11] = _mm_sub_epi32(y, x);
        u[11] = _mm_add_epi32(u[11], rnding);
        u[11] = _mm_srai_epi32(u[11], bit);

        y = _mm_mullo_epi32(u[14], cospi32);
        x = _mm_mullo_epi32(u[15], cospi32);
        u[14] = _mm_add_epi32(y, x);
        u[14] = _mm_add_epi32(u[14], rnding);
        u[14] = _mm_srai_epi32(u[14], bit);

        u[15] = _mm_sub_epi32(y, x);
        u[15] = _mm_add_epi32(u[15], rnding);
        u[15] = _mm_srai_epi32(u[15], bit);

        // stage 9
        if (do_cols) {
            out[0] = u[0];
            out[1] = _mm_sub_epi32(_mm_setzero_si128(), u[8]);
            out[2] = u[12];
            out[3] = _mm_sub_epi32(_mm_setzero_si128(), u[4]);
            out[4] = u[6];
            out[5] = _mm_sub_epi32(_mm_setzero_si128(), u[14]);
            out[6] = u[10];
            out[7] = _mm_sub_epi32(_mm_setzero_si128(), u[2]);
            out[8] = u[3];
            out[9] = _mm_sub_epi32(_mm_setzero_si128(), u[11]);
            out[10] = u[15];
            out[11] = _mm_sub_epi32(_mm_setzero_si128(), u[7]);
            out[12] = u[5];
            out[13] = _mm_sub_epi32(_mm_setzero_si128(), u[13]);
            out[14] = u[9];
            out[15] = _mm_sub_epi32(_mm_setzero_si128(), u[1]);
        }
        else {
            const int32_t log_range_out = AOMMAX(16, bd + 6);
            const __m128i clamp_lo_out = _mm_set1_epi32(-(1 << (log_range_out - 1)));
            const __m128i clamp_hi_out =
                _mm_set1_epi32((1 << (log_range_out - 1)) - 1);

            neg_shift_sse4_1(u[0], u[8], out + 0, out + 1, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[12], u[4], out + 2, out + 3, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[6], u[14], out + 4, out + 5, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[10], u[2], out + 6, out + 7, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[3], u[11], out + 8, out + 9, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[15], u[7], out + 10, out + 11, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[5], u[13], out + 12, out + 13, &clamp_lo_out,
                &clamp_hi_out, out_shift);
            neg_shift_sse4_1(u[9], u[1], out + 14, out + 15, &clamp_lo_out,
                &clamp_hi_out, out_shift);
        }
    }
}

static INLINE void highbd_write_buffer_4xn_sse4_1(__m128i *in, uint16_t *output,
    int32_t stride, int32_t flipud,
    int32_t height, const int32_t bd) {
    int32_t j = flipud ? (height - 1) : 0;
    const int32_t step = flipud ? -1 : 1;
    for (int32_t i = 0; i < height; ++i, j += step) {
        __m128i v = _mm_loadl_epi64((__m128i const *)(output + i * stride));
        __m128i u = highbd_get_recon_4xn_sse4_1(v, in[j], bd);

        _mm_storel_epi64((__m128i *)(output + i * stride), u);
    }
}

static const transform_1d_sse4_1
highbd_txfm_all_1d_zeros_w8_arr[TX_SIZES][ITX_TYPES_1D][4] = {
    {
        { idct4x4_sse4_1, NULL, NULL, NULL },
        { iadst4x4_sse4_1, NULL, NULL, NULL },
        { iidentity4_sse4_1, iidentity4_sse4_1, iidentity4_sse4_1, NULL },
    },
    { { idct8x8_low1_sse4_1, idct8x8_new_sse4_1, NULL, NULL },
    { iadst8x8_low1_sse4_1, iadst8x8_new_sse4_1, NULL, NULL },
    { iidentity8_sse4_1, iidentity8_sse4_1, NULL, NULL } },
    {
        { idct16x16_low1_sse4_1, idct16x16_low8_sse4_1, idct16x16_new_sse4_1,
        NULL },
        { iadst16x16_low1_sse4_1, iadst16x16_low8_sse4_1, iadst16x16_new_sse4_1,
        NULL },
        { iidentity16_sse4_1, NULL, iidentity16_sse4_1, NULL },
    },
    { { NULL, NULL, NULL,
    NULL },
    { NULL, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL } },
    { { NULL, NULL, NULL,
    NULL },
    { NULL, NULL, NULL, NULL },
    { NULL, NULL, NULL, NULL } }
};

void av1_inv_txfm2d_add_4x8_sse4_1(const int32_t *input,
    uint16_t *output, int32_t stride,
    TxType tx_type, TxSize tx_size, int32_t bd) {
    __m128i buf1[8];
    const int8_t *shift = inv_txfm_shift_ls[tx_size];
    const int32_t txw_idx = get_txw_idx(tx_size);
    const int32_t txh_idx = get_txh_idx(tx_size);
    const int32_t txfm_size_col = tx_size_wide[tx_size];
    const int32_t txfm_size_row = tx_size_high[tx_size];
    const transform_1d_sse4_1 row_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txw_idx][hitx_1d_tab[tx_type]][0];
    const transform_1d_sse4_1 col_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txh_idx][vitx_1d_tab[tx_type]][1];
    const int32_t input_stride = AOMMIN(32, txfm_size_col);

    assert(col_txfm != NULL);
    assert(row_txfm != NULL);
    int32_t ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);

    // 1st stage: column transform
    __m128i buf0[8];
    const int32_t *input_row = input;
    __m128i *buf0_cur = buf0;
    load_buffer_32bit_input(input_row, input_stride, buf0_cur, txfm_size_row);
    av1_round_shift_rect_array_32_sse4_1(buf0, buf0, txfm_size_row, 0,
        NewInvSqrt2);
    row_txfm(buf0, buf0, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, -shift[0]);
    row_txfm(buf0 + 4, buf0 + 4, inv_cos_bit_row[txw_idx][txh_idx], 0, bd,
        -shift[0]);

    if (lr_flip) {
        TRANSPOSE_4X4(buf0[3], buf0[2], buf0[1], buf0[0], buf1[0], buf1[1], buf1[2],
            buf1[3]);

        TRANSPOSE_4X4(buf0[7], buf0[6], buf0[5], buf0[4], buf1[4], buf1[5], buf1[6],
            buf1[7]);
    }
    else {
        TRANSPOSE_4X4(buf0[0], buf0[1], buf0[2], buf0[3], buf1[0], buf1[1], buf1[2],
            buf1[3]);

        TRANSPOSE_4X4(buf0[4], buf0[5], buf0[6], buf0[7], buf1[4], buf1[5], buf1[6],
            buf1[7]);
    }

    // 2nd stage: column transform
    col_txfm(buf1, buf1, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);

    av1_round_shift_array_32_sse4_1(buf1, buf1, txfm_size_row, -shift[1]);

    // write to buffer
    highbd_write_buffer_4xn_sse4_1(buf1, output, stride, ud_flip, txfm_size_row,
        bd);
}

//8x4
static INLINE void flip_buf_sse2(__m128i *in, __m128i *out, int32_t size) {
    for (int32_t i = 0; i < size; ++i)
        out[size - i - 1] = in[i];
}

static INLINE __m128i highbd_get_recon_8x8_sse4_1(const __m128i pred,
    __m128i res0, __m128i res1,
    const int32_t bd) {
    __m128i x0 = _mm_cvtepi16_epi32(pred);
    __m128i x1 = _mm_cvtepi16_epi32(_mm_srli_si128(pred, 8));

    x0 = _mm_add_epi32(res0, x0);
    x1 = _mm_add_epi32(res1, x1);
    x0 = _mm_packus_epi32(x0, x1);
    x0 = highbd_clamp_epi16(x0, bd);
    return x0;
}

static INLINE void highbd_write_buffer_8xn_sse4_1(__m128i *in, uint16_t *output,
    int32_t stride, int32_t flipud,
    int32_t height, const int32_t bd) {
    int32_t j = flipud ? (height - 1) : 0;
    const int32_t step = flipud ? -1 : 1;
    for (int32_t i = 0; i < height; ++i, j += step) {
        __m128i v = _mm_loadu_si128((__m128i const *)(output + i * stride));
        __m128i u = highbd_get_recon_8x8_sse4_1(v, in[j], in[j + height], bd);

        _mm_storeu_si128((__m128i *)(output + i * stride), u);
    }
}
void av1_inv_txfm2d_add_8x4_sse4_1(const int32_t *input,
    uint16_t *output, int32_t stride,
    TxType tx_type, TxSize tx_size, int32_t bd) {
    __m128i buf1[8];
    const int8_t *shift = inv_txfm_shift_ls[tx_size];
    const int32_t txw_idx = get_txw_idx(tx_size);
    const int32_t txh_idx = get_txh_idx(tx_size);
    const int32_t txfm_size_col = tx_size_wide[tx_size];
    const int32_t txfm_size_row = tx_size_high[tx_size];
    const transform_1d_sse4_1 row_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txw_idx][hitx_1d_tab[tx_type]][1];
    const transform_1d_sse4_1 col_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txh_idx][vitx_1d_tab[tx_type]][0];

    assert(col_txfm != NULL);
    assert(row_txfm != NULL);
    int32_t ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);

    // 1st stage: column transform
    __m128i buf0[8];
    const int32_t *input_row = input;
    load_buffer_32bit_input(input_row, 4, buf0, txfm_size_col);

    TRANSPOSE_4X4(buf0[0], buf0[2], buf0[4], buf0[6], buf1[0], buf1[1], buf1[2],
        buf1[3]);
    TRANSPOSE_4X4(buf0[1], buf0[3], buf0[5], buf0[7], buf1[4], buf1[5], buf1[6],
        buf1[7]);

    av1_round_shift_rect_array_32_sse4_1(buf1, buf0, txfm_size_col, 0,
        NewInvSqrt2);
    row_txfm(buf0, buf0, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, -shift[0]);

    __m128i *buf1_ptr;
    if (lr_flip) {
        flip_buf_sse2(buf0, buf1, txfm_size_col);
        buf1_ptr = buf1;
    }
    else
        buf1_ptr = buf0;
    // 2nd stage: column transform
    for (int32_t i = 0; i < 2; i++) {
        col_txfm(buf1_ptr + i * txfm_size_row, buf1_ptr + i * txfm_size_row,
            inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
    }
    av1_round_shift_array_32_sse4_1(buf1_ptr, buf1_ptr, txfm_size_col, -shift[1]);
    // write to buffer
    highbd_write_buffer_8xn_sse4_1(buf1_ptr, output, stride, ud_flip,
        txfm_size_row, bd);
}

//4x16
void av1_inv_txfm2d_add_4x16_sse4_1(const int32_t *input,
    uint16_t *output, int32_t stride,
    TxType tx_type, TxSize tx_size,int32_t bd) {
    __m128i buf1[16];
    const int8_t *shift = inv_txfm_shift_ls[tx_size];
    const int32_t txw_idx = get_txw_idx(tx_size);
    const int32_t txh_idx = get_txh_idx(tx_size);
    const int32_t txfm_size_col = tx_size_wide[tx_size];
    const int32_t txfm_size_row = tx_size_high[tx_size];
    const int32_t buf_size_h_div8 = txfm_size_row >> 2;
    const transform_1d_sse4_1 row_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txw_idx][hitx_1d_tab[tx_type]][0];
    const transform_1d_sse4_1 col_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txh_idx][vitx_1d_tab[tx_type]][2];
    const int32_t input_stride = AOMMIN(32, txfm_size_col);

    assert(col_txfm != NULL);
    assert(row_txfm != NULL);
    int32_t ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);

    // 1st stage: column transform
    __m128i buf0[16];
    const int32_t *input_row = input;
    __m128i *buf0_cur = buf0;
    load_buffer_32bit_input(input_row, input_stride, buf0_cur, txfm_size_row);
    for (int32_t i = 0; i < (txfm_size_row >> 2); i++) {
        row_txfm(buf0 + (i << 2), buf0 + (i << 2),
            inv_cos_bit_row[txw_idx][txh_idx], 0, bd, -shift[0]);
    }

    av1_round_shift_array_32_sse4_1(buf0, buf0, txfm_size_row, -shift[0]);

    if (lr_flip) {
        for (int32_t j = 0; j < buf_size_h_div8; ++j) {
            TRANSPOSE_4X4(buf0[4 * j + 3], buf0[4 * j + 2], buf0[4 * j + 1],
                buf0[4 * j], buf1[4 * j], buf1[4 * j + 1], buf1[4 * j + 2],
                buf1[4 * j + 3]);
        }
    }
    else {
        for (int32_t j = 0; j < buf_size_h_div8; ++j) {
            TRANSPOSE_4X4(buf0[4 * j], buf0[4 * j + 1], buf0[4 * j + 2],
                buf0[4 * j + 3], buf1[4 * j], buf1[4 * j + 1],
                buf1[4 * j + 2], buf1[4 * j + 3]);
        }
    }

    // 2nd stage: column transform
    col_txfm(buf1, buf1, inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);

    av1_round_shift_array_32_sse4_1(buf1, buf1, txfm_size_row, -shift[1]);

    // write to buffer
    highbd_write_buffer_4xn_sse4_1(buf1, output, stride, ud_flip, txfm_size_row,
        bd);
}

//16x4
void av1_inv_txfm2d_add_16x4_sse4_1(const int32_t *input,
    uint16_t *output, int32_t stride,
    TxType tx_type, TxSize tx_size, int32_t bd) {
    __m128i buf1[16];
    const int8_t *shift = inv_txfm_shift_ls[tx_size];
    const int32_t txw_idx = get_txw_idx(tx_size);
    const int32_t txh_idx = get_txh_idx(tx_size);
    const int32_t txfm_size_col = tx_size_wide[tx_size];
    const int32_t txfm_size_row = tx_size_high[tx_size];
    const int32_t buf_size_w_div8 = txfm_size_col >> 2;
    const transform_1d_sse4_1 row_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txw_idx][hitx_1d_tab[tx_type]][2];
    const transform_1d_sse4_1 col_txfm =
        highbd_txfm_all_1d_zeros_w8_arr[txh_idx][vitx_1d_tab[tx_type]][0];

    assert(col_txfm != NULL);
    assert(row_txfm != NULL);
    int32_t ud_flip, lr_flip;
    get_flip_cfg(tx_type, &ud_flip, &lr_flip);

    // 1st stage: column transform
    __m128i buf0[16];
    const int32_t *input_row = input;
    load_buffer_32bit_input(input_row, 4, buf0, txfm_size_col);

    for (int32_t j = 0; j < buf_size_w_div8; j++) {
        TRANSPOSE_4X4(buf0[j], buf0[j + 4], buf0[j + 8], buf0[j + 12], buf1[4 * j],
            buf1[4 * j + 1], buf1[4 * j + 2], buf1[4 * j + 3]);
    }
    row_txfm(buf1, buf0, inv_cos_bit_row[txw_idx][txh_idx], 0, bd, -shift[0]);

    __m128i *buf1_ptr;
    if (lr_flip) {
        flip_buf_sse2(buf0, buf1, txfm_size_col);
        buf1_ptr = buf1;
    }
    else
        buf1_ptr = buf0;
    // 2nd stage: column transform
    for (int32_t i = 0; i < buf_size_w_div8; i++) {
        col_txfm(buf1_ptr + i * txfm_size_row, buf1_ptr + i * txfm_size_row,
            inv_cos_bit_col[txw_idx][txh_idx], 1, bd, 0);
    }
    av1_round_shift_array_32_sse4_1(buf1_ptr, buf1_ptr, txfm_size_col, -shift[1]);

    // write to buffer
    for (int32_t i = 0; i < (txfm_size_col >> 3); i++) {
        highbd_write_buffer_8xn_sse4_1(buf1_ptr + i * txfm_size_row * 2,
            output + 8 * i, stride, ud_flip,
            txfm_size_row, bd);
    }
}
