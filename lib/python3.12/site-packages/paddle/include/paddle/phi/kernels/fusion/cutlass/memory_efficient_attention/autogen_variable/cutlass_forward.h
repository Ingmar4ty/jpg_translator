// This file is auto-generated. See "generate_variable_forward_kernels.py"
#pragma once
#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen_variable/memory_efficient_variable_attention.h"
namespace phi {
// ======== bf16 / sm80 ========


void  fmha_cutlassF_variable_bf16_aligned_32x128_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_32x128_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_64x64_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_64x64_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_32x128_urf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_32x128_urf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_32x128_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_64x64_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_bf16_aligned_32x128_urf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_bf16_sm80(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_32x128_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_32x128_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_64x64_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_64x64_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_32x128_urf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_bf16_aligned_32x128_urf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_bf16_aligned_32x128_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_bf16_aligned_64x64_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::bfloat16_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_bf16_aligned_32x128_urf_usm_mua_sm80);
}

// ======== f16 / sm50 ========


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f16_sm50(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm50);
}

// ======== f16 / sm70 ========


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f16_sm70(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm70);
}

// ======== f16 / sm75 ========


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f16_sm75(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_64x64_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_notaligned_32x128_urf_usm_mua_sm75);
}

// ======== f16 / sm80 ========


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f16_sm80(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_64x64_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<cutlass::half_t, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f16_aligned_32x128_urf_usm_mua_sm80);
}

// ======== f32 / sm50 ========


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm50(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f32_sm50(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm50);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm50, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm50);
}

// ======== f32 / sm70 ========


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm70(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f32_sm70(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm70);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm70, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm70);
}

// ======== f32 / sm75 ========


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm75(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f32_sm75(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_ma_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_sm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_64x64_rf_usm_mua_sm75);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm75, false, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_notaligned_32x128_urf_usm_mua_sm75);
}

// ======== f32 / sm80 ========


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);


void  fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm80(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false> default_fmha, Params &params, const phi::GPUContext& ctx);

template <typename T> void dispatch_cutlass_forward_f32_sm80(T cb) {
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, true, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_ma_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, true>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_sm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 64, 64, true, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_64x64_rf_usm_mua_sm80);
    cb(cutlass::gemm::kernel::DefaultFMHAGrouped<float, cutlass::arch::Sm80, true, false, 32, 128, false, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, false>(), fmha_cutlassF_variable_f32_aligned_32x128_urf_usm_mua_sm80);
}


template <typename PaddleT, typename T>
void dispatch_cutlass_forward(const ::phi::GPUContext &ctx, T cb) {
    auto cc = ctx.GetComputeCapability();
    PADDLE_ENFORCE_GE(
        cc,
        70,
        phi::errors::InvalidArgument("the Nvidia GPU's Compute Capability must be greater or equal than 70"));

    using DT = typename ::phi::CutlassTrait<PaddleT>::Type;

    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 90) {
        dispatch_cutlass_forward_bf16_sm80(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 50 <= cc && cc < 70) {
        dispatch_cutlass_forward_f16_sm50(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 70 <= cc && cc < 75) {
        dispatch_cutlass_forward_f16_sm70(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 75 <= cc && cc < 80) {
        dispatch_cutlass_forward_f16_sm75(cb);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 90) {
        dispatch_cutlass_forward_f16_sm80(cb);
    }
    if (std::is_same<DT, float>::value && 50 <= cc && cc < 70) {
        dispatch_cutlass_forward_f32_sm50(cb);
    }
    if (std::is_same<DT, float>::value && 70 <= cc && cc < 75) {
        dispatch_cutlass_forward_f32_sm70(cb);
    }
    if (std::is_same<DT, float>::value && 75 <= cc && cc < 80) {
        dispatch_cutlass_forward_f32_sm75(cb);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 90) {
        dispatch_cutlass_forward_f32_sm80(cb);
    }
}
} // namespace phi
#endif // PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION
