# Collaborative Cache Library

## User guide

The main interface of this library is the `build_v2` & `lookup` method of class `CollCache` in `facade.h`.

To use this library, follow these steps:

- inherit these class:
  - `ExternelGPUMemoryHandler`
    - handler of gpu memory allocated by outside application. when deconstructed, the memory should be freed automatically.
    - example:
      ```c++
      class DemoMemHandle : public ExternelGPUMemoryHandler {
      public:
        void* dev_ptr = nullptr;
        void* ptr() override {return dev_ptr;}
        ~DemoMemHandle() { CUDA_CALL(cudaFree(dev_ptr)); }
      };
      ```
  - `ExternalBarrierHandler`
    - handler of global barrier provided by outside application.
    - example:
      ```c++
      // intra-process
      class DemoBarrier : public ExternalBarrierHandler {
      public:
        AtomicBarrier barrier;
        DemoBarrier(int worker) : barrier(worker) {}
        void Wait() override { barrier.Wait(); }
      };
      // inter-process. the constructor must be called before fork
      class DemoBarrier : public ExternalBarrierHandler {
      public:
        AtomicBarrier* barrier;
        DemoBarrier(int worker) {
          auto mmap_ctx = MMAP(MMAP_RW_DEVICE);
          auto dev = Device::Get(mmap_ctx);
          barrier = new(dev->AllocArray<AtomicBarrier>(mmap_ctx, 1)) AtomicBarrier(worker);
        }
        void Wait() override { barrier->Wait(); }
      };
      ```
- intialize necessary member of `RunConfig`
- created a `shared_ptr`-managed instance of `CollCache`
- call `build_v2` concurrently.
  - this can be done inter- or intra-process depending on application.
- for each replica, use `lookup` for extraction.