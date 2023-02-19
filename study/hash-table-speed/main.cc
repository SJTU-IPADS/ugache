#include "cuda/cuda_hashtable.h"
#include "cpu/cpu_utils.h"
#include "device.h"
#include "timer.h"
#include <CLI/CLI.hpp>
using namespace samgraph;
using namespace samgraph::common;

IdType num_full_key = 800000000;
double cache_rate = 0.05;
double evict_rate = 0.01;
double hit_rate  = 0.95;
IdType rqst_size = 8000*100;
IdType num_repeat_lookup = 10;
IdType num_refresh = 10;

void shuffle_first(IdType* array, size_t num_to_shuffle, size_t array_len) {
  auto g = std::default_random_engine(0x123455);
  for (size_t idx = 0; idx < num_to_shuffle; idx++) {
    std::uniform_int_distribution<size_t> d(idx, array_len - 1);
    size_t candidate = d(g);
    std::swap(array[candidate], array[idx]);
  }
}

TensorPtr subtensor(TensorPtr from, size_t begin, size_t end, StreamHandle stream) {
  CHECK(from->Shape().size() == 1);
  CHECK(end > begin);
  CHECK(from->Shape()[0] >= end);
  return Tensor::CopyBlob((uint8_t*)from->Data() + begin * GetDataTypeBytes(from->Type()), from->Type(), {end-begin}, from->Ctx(), from->Ctx(), "", stream);
}
TensorPtr subtensor(TensorPtr from, size_t begin, size_t end, Context dst_ctx, StreamHandle stream) {
  return Tensor::CopyTo(subtensor(from, begin, end, stream), dst_ctx, stream);
}

TensorPtr gen_rnd_rqst(const IdType* src, const size_t num_src, const size_t num_rqst, Context dst_ctx, StreamHandle stream = nullptr) {
  TensorPtr to = Tensor::Empty(kI32, {num_rqst}, CPU(), "");
  thread_local std::mt19937 g;
  #pragma omp parallel for
  for (size_t i = 0; i < to->Shape()[0]; i++) {
    std::uniform_int_distribution<size_t> d(0, num_src - 1);
    to->Ptr<IdType>()[i] = src[d(g)];
  }
  if (dst_ctx == CPU()) {
    return to;
  }
  return Tensor::CopyTo(to, dst_ctx, stream);
}

int main(int argc, char** argv) {
  CLI::App _app;
  // _app.add_option("--arch", configs["arch"])
  //     ->check(CLI::IsMember({
  //         "arch0",
  //         "arch1",
  //         "arch2",
  //         "arch3",
  //     }));
  _app.add_option("--nkey", num_full_key);
  _app.add_option("--cache-rate,-c", cache_rate);
  _app.add_option("--evict-rate,-e", evict_rate);
  _app.add_option("--hit-rate,--hit", hit_rate);
  _app.add_option("--batch-size,-b", rqst_size);
  _app.add_option("--nr", num_refresh);
  _app.add_option("--nl", num_repeat_lookup);

  try {
    _app.parse(argc, argv);
  } catch(const CLI::ParseError &e) {
    _app.exit(e);
    exit(1);
  }
  IdType cached_keys  = num_full_key * cache_rate;
  IdType rqst_range = cached_keys / hit_rate;

  StreamHandle stream;
  CUDA_CALL(cudaStreamCreate((cudaStream_t*)&stream));
  Context gpu_ctx = GPU(0);
  Device* gpu_dev = Device::Get(gpu_ctx);
  auto hash_table = std::make_shared<cuda::OrderedHashTable>(cached_keys, gpu_ctx, stream);
  IdType num_evict_keys = cached_keys * evict_rate;
  TensorPtr cpu_total_key_list = Tensor::Empty(kI32, {num_full_key}, CPU(), "");

  LOG(ERROR) << "preparing full key array";
  cpu::ArrangeArray(cpu_total_key_list->Ptr<IdType>(), num_full_key);
  IdType* non_cached_keys_cpu = cpu_total_key_list->Ptr<IdType>() + cached_keys;
  IdType* cached_keys_cpu = cpu_total_key_list->Ptr<IdType>();

  LOG(ERROR) << "preparing random cache keys";
  shuffle_first(cpu_total_key_list->Ptr<IdType>(), cached_keys + num_evict_keys, num_full_key);
  TensorPtr gpu_cached_nodes = subtensor(cpu_total_key_list, 0, cached_keys, gpu_ctx, stream);
  gpu_dev->StreamSync(gpu_ctx, stream);
  LOG(ERROR) << "filling hash table";
  hash_table->FillWithUnique(gpu_cached_nodes->CPtr<IdType>(), cached_keys, stream);
  gpu_dev->StreamSync(gpu_ctx, stream);
  auto lookup_rst = Tensor::Empty(kI32, {rqst_size}, gpu_ctx, "");
  auto tmp_lookup_rst = Tensor::Empty(kI32, {num_evict_keys}, gpu_ctx, "");

  auto gen_rqst = [&cpu_total_key_list, gpu_ctx, stream, gpu_dev, rqst_range](){
    auto rqst = gen_rnd_rqst(cpu_total_key_list->CPtr<IdType>(), rqst_range, rqst_size, gpu_ctx, stream);
    gpu_dev->StreamSync(gpu_ctx, stream);
    return rqst;
  };
  auto lookup = [hash_table, lookup_rst, stream, gpu_dev, gpu_ctx](TensorPtr rqst){
    Timer t;
    hash_table->LookupIfExist(rqst->CPtr<IdType>(), rqst_size, lookup_rst->Ptr<IdType>(), stream);
    gpu_dev->StreamSync(gpu_ctx, stream);
    LOG(ERROR) << "lookup time " << t.PassedMicro();
  };
  auto evict = [cpu_total_key_list, gpu_ctx, stream, gpu_dev, hash_table, cached_keys, tmp_lookup_rst](){
    IdType num_evict_keys = cached_keys * evict_rate;
    IdType* non_cached_keys_cpu = cpu_total_key_list->Ptr<IdType>() + cached_keys;
    IdType* cached_keys_cpu = cpu_total_key_list->Ptr<IdType>();

    shuffle_first(cached_keys_cpu, num_evict_keys, cached_keys);
    shuffle_first(non_cached_keys_cpu, num_evict_keys, num_full_key - cached_keys);

    TensorPtr keys_to_evict = Tensor::CopyBlob(cached_keys_cpu, kI32, {num_evict_keys}, CPU(), gpu_ctx, "", stream);
    TensorPtr new_keys = Tensor::CopyBlob(non_cached_keys_cpu, kI32, {num_evict_keys}, CPU(), gpu_ctx, "", stream);
    gpu_dev->CopyDataFromTo(keys_to_evict->Data(), 0, non_cached_keys_cpu, 0, keys_to_evict->NumBytes(), gpu_ctx, CPU(), stream);
    gpu_dev->CopyDataFromTo(new_keys->Data(), 0, cached_keys_cpu, 0, new_keys->NumBytes(), gpu_ctx, CPU(), stream);
    gpu_dev->StreamSync(gpu_ctx, stream);

    Timer t;
    hash_table->EvictWithUnique(keys_to_evict->CPtr<IdType>(), num_evict_keys, stream);
    hash_table->FillWithUnique(new_keys->CPtr<IdType>(), num_evict_keys, stream);
    gpu_dev->StreamSync(gpu_ctx, stream);
    LOG(ERROR) << "evict time " << t.PassedMicro();

    hash_table->LookupIfExist(keys_to_evict->CPtr<IdType>(), num_evict_keys, tmp_lookup_rst->Ptr<IdType>(), stream);
    cuda::check_cuda_array(tmp_lookup_rst->Ptr<IdType>(), cuda::kEmptyPos, num_evict_keys, true, stream);
    gpu_dev->StreamSync(gpu_ctx, stream);
    hash_table->LookupIfExist(new_keys->CPtr<IdType>(), num_evict_keys, tmp_lookup_rst->Ptr<IdType>(), stream);
    cuda::check_cuda_array(tmp_lookup_rst->Ptr<IdType>(), cuda::kEmptyPos, num_evict_keys, false, stream);
    gpu_dev->StreamSync(gpu_ctx, stream);

    // shuffle_first(cached_keys_cpu, num_evict_keys, cached_keys);
    // shuffle_first(non_cached_keys_cpu, num_evict_keys, num_full_key - cached_keys);
  };
  hash_table->CountEntries(stream);

  for (IdType i = 0; i < num_refresh; i++) {
    for (IdType j = 0; j < num_repeat_lookup; j++) {
      lookup(gen_rqst());
    }
    evict();
    hash_table->CountEntries(stream);
  }
  for (IdType j = 0; j < num_repeat_lookup; j++) {
    lookup(gen_rqst());
  }

}