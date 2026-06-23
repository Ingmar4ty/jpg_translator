
#pragma once

#ifdef PADDLE_WITH_MEMORY_EFFICIENT_ATTENTION

#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/kernel_forward.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/kernel_backward.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {

template <typename T>
struct CutlassTrait {
  using Type = T;
};

template <>
struct CutlassTrait<dtype::float16> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassTrait<dtype::bfloat16> {
  using Type = cutlass::bfloat16_t;
};


template <typename T>
struct ToPhiDTypeTrait {
 private:
  using NonConstT = typename std::remove_const<T>::type;
  static constexpr bool kIsFP16 = std::is_same<NonConstT, cutlass::half_t>::value;
  static constexpr bool kIsBF16 = std::is_same<NonConstT, cutlass::bfloat16_t>::value;

 public:
  using Type = typename std::conditional<kIsFP16, dtype::float16,
      typename std::conditional<kIsBF16, dtype::bfloat16, NonConstT>::type>::type;
};


template <typename T>
T *SafeGetTensorPtr(const DenseTensor &t) {
  using PDT = typename ToPhiDTypeTrait<T>::Type;
  return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(t.template data<PDT>()));
}

template <typename T>
T *SafeGetTensorPtr(const DenseTensor *t) {
  return t ? SafeGetTensorPtr<T>(*t) : nullptr;
}

template <typename T>
T *SafeGetTensorPtr(const paddle::optional<DenseTensor> &t) {
  return t ? SafeGetTensorPtr<T>(t.get()) : nullptr;
}

template <typename T, typename Context>
T *SafeAllocTensor(const Context &ctx, DenseTensor *t) {
  using PDT = typename ToPhiDTypeTrait<T>::Type;
  void *ptr = ctx.template Alloc<PDT>(t);
  return reinterpret_cast<T *>(reinterpret_cast<uintptr_t>(ptr));
}

inline int64_t DimStride(const phi::DDim &dims, int n) {
  int rank = dims.size();
  if (n < 0) {
    n += rank;
  }
  int64_t stride = 1;
  for (int i = n+1; i < rank; ++i) {
    stride *= dims[i];
  }
  return stride;
}

} // namespace phi

#include "./cutlass_forward.h"
#include "./cutlass_backward.h"

#endif
