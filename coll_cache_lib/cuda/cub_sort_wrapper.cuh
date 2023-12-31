#pragma once
#include "../common.h"
#include "../device.h"
// #include "../timer.h"
#include "../logging.h"
#include <cub/cub.cuh>
namespace coll_cache_lib{
namespace common{
namespace cuda{

template<typename T>
struct SelectIdxByEqual {
  const T* d_array;
  T compare;
  CUB_RUNTIME_FUNCTION __forceinline__
  SelectIdxByEqual(const T* d_array, const T compare) : d_array(d_array), compare(compare) {}
  __host__ __device__  __forceinline__
  bool operator()(const T & idx) const {
    return d_array[idx] == compare;
  }
};
template<typename T>
struct SelectIdxByNotEqual {
  const T* d_array;
  T compare;
  CUB_RUNTIME_FUNCTION __forceinline__
  SelectIdxByNotEqual(const T* d_array, const T compare) : d_array(d_array), compare(compare) {}
  __host__ __device__  __forceinline__
  bool operator()(const T & idx) const {
    return d_array[idx] != compare;
  }
};

template<typename T>
void CubSelectIndexByEq(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    T* d_out, size_t & num_selected_out,
    const T compare,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::CountingInputIterator<T> counter(0);
  SelectIdxByEqual<T> select_op(d_in, compare);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}

template<typename T, typename T_Side>
void CubSelectIndexByEqSide(Context gpu_ctx,
    const T_Side * d_in, const size_t num_input,
    T* d_out, size_t & num_selected_out,
    const T_Side compare,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::CountingInputIterator<T> counter(0);
  SelectIdxByEqual<T_Side> select_op(d_in, compare);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  CUDA_CALL(cudaStreamSynchronize(cu_stream));
  // device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}

template<typename T, typename T_Side>
void CubSelectBySideEq(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    const T_Side * d_side,
    T* d_out, size_t & num_selected_out,
    const T_Side compare,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  SelectIdxByEqual<T_Side> select_op(d_side, compare);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}
template<typename T, typename T_Side>
void CubSelectBySideNe(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    const T_Side * d_side,
    T* d_out, size_t & num_selected_out,
    const T_Side compare,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  SelectIdxByNotEqual<T_Side> select_op(d_side, compare);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}

template<typename T, typename SelectOp>
void CubSelectIndex(Context gpu_ctx,
    const size_t num_input,
    T* d_out, size_t & num_selected_out,
    SelectOp select_op,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::CountingInputIterator<T> counter(0);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, counter, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  CUDA_CALL(cudaStreamSynchronize(cu_stream));
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}
template<typename T, typename SelectOp>
void CubSelect(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    T* d_out, size_t & num_selected_out,
    SelectOp select_op,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t * d_num_selected_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceSelect::If(workspace, workspace_bytes, d_in, d_out, d_num_selected_out, num_input, select_op, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  num_selected_out = *d_num_selected_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_num_selected_out);
}

template<typename T>
struct EqConverter {
  T compare;
  __host__ __device__ __forceinline__
  EqConverter(const T & compare) : compare(compare) {}
  __host__ __device__ __forceinline__
  size_t operator()(const T & a) const {
    return (a == compare) ? 1 : 0;
  }
};

template<typename T>
void CubCountByEq(Context gpu_ctx,
    const T * d_in, const size_t num_input,
    size_t & count_out,
    const T compare,
    std::function<MemHandle(size_t)> & allocator,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  EqConverter<T> converter(compare);
  cub::TransformInputIterator<size_t, EqConverter<T>, T*> itr(const_cast<T*>(d_in), converter);

  size_t * d_count_out = Device::Get(CPU())->AllocArray<size_t>(CPU(), 1);

  size_t workspace_bytes;
  void * workspace = nullptr;
  cub::DeviceReduce::Sum(workspace, workspace_bytes, itr, d_count_out, num_input, cu_stream);
  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  cub::DeviceReduce::Sum(workspace, workspace_bytes, itr, d_count_out, num_input, cu_stream);
  device->StreamSync(gpu_ctx, stream);
  count_out = *d_count_out;
  Device::Get(CPU())->FreeWorkspace(CPU(), d_count_out);
}

template<typename NativeKey_t>
void CubSortKeyInPlace(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  key_alter = keys.Alternate();
}

template<typename NativeKey_t>
void CubSortKey(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const size_t num_nodes, Context gpu_ctx,
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortKeys(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

}

template<typename NativeKey_t>
void CubSortKeyDescendingInplace(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, keys, len,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  key_alter = keys.Alternate();

}

template<typename NativeKey_t>
void CubSortKeyDescending(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const size_t num_nodes, Context gpu_ctx,
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortKeysDescending(
      workspace, workspace_bytes, key_in, key_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortPairInplace(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    NativeVal_t* & val, NativeVal_t* & val_alter,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);
  cub::DoubleBuffer<NativeVal_t> vals(val, val_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  val = vals.Current();
  key_alter = keys.Alternate();
  val_alter = vals.Alternate();

}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortPair(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const NativeVal_t* val_in, NativeVal_t* val_out,
    const size_t num_nodes, Context gpu_ctx,
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairs(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortPairDescendingInplace(
    NativeKey_t* & key, NativeKey_t* & key_alter,
    NativeVal_t* & val, NativeVal_t* & val_alter,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream = nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  cub::DoubleBuffer<NativeKey_t> keys(key, key_alter);
  cub::DoubleBuffer<NativeVal_t> vals(val, val_alter);

  size_t workspace_bytes;
  void * workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace, workspace_bytes, keys, vals, len,
      begin_bit, end_bit, cu_stream, false));
  device->StreamSync(gpu_ctx, stream);

  key = keys.Current();
  val = vals.Current();
  key_alter = keys.Alternate();
  val_alter = vals.Alternate();

}
template<typename NativeKey_t, typename NativeVal_t>
void CubSortPairDescending(
    const NativeKey_t* key_in, NativeKey_t* key_out,
    const NativeVal_t* val_in, NativeVal_t* val_out,
    const size_t num_nodes, Context gpu_ctx,
    std::function<MemHandle(size_t)> & allocator,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8,
    StreamHandle stream=nullptr) {
  auto cu_stream = static_cast<cudaStream_t>(stream);
  auto device = Device::Get(gpu_ctx);

  size_t workspace_bytes;
  void *workspace = nullptr;
  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));

  auto workspace_mem_handle = allocator(workspace_bytes);
  workspace = workspace_mem_handle->ptr();

  CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
      workspace, workspace_bytes, key_in, key_out, val_in, val_out, num_nodes,
      begin_bit, end_bit, cu_stream));
  device->StreamSync(gpu_ctx, stream);

}

template<typename NativeKey_t, typename NativeVal_t>
void CubSortDispatcher(
    NativeKey_t* & key, NativeKey_t* & key_out,
    NativeVal_t* & val, NativeVal_t* & val_out,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    bool decending = false,
    StreamHandle stream = nullptr,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8) {
  if (decending == false) {
    CubSortPairInplace(key, key_out, val, val_out, len, gpu_ctx, allocator, begin_bit, end_bit, stream);
  } else {
    CubSortPairDescendingInplace(key, key_out, val, val_out, len, gpu_ctx, allocator, begin_bit, end_bit, stream);
  }
}
template<typename NativeKey_t>
void CubSortDispatcher(
    NativeKey_t* & key, NativeKey_t* & key_out,
    void* _1, void* _2,
    const size_t len, Context gpu_ctx, 
    std::function<MemHandle(size_t)> & allocator,
    bool decending = false,
    StreamHandle stream = nullptr,
    int begin_bit = 0, int end_bit = sizeof(NativeKey_t) * 8) {
  if (decending == false) {
    CubSortKeyInPlace(key, key_out, len, gpu_ctx, allocator, begin_bit, end_bit, stream);
  } else {
    CubSortKeyDescendingInplace(key, key_out, len, gpu_ctx, allocator, begin_bit, end_bit, stream);
  }
}

}
}
}
