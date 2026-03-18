#pragma once

#include "context.cuh"

// This queue's capacity and runtime behavior is bounded by the numebr of slots
// each element is associated with one or more slot
// So if the number of slots > QSIZE, both enqueue and dequeue will be safe,
// reads as the slots are guaranteed to be unique per element
template<typename T, unsigned QSIZE = 32>
struct SizeBoundedBarrierQueue {
  static_assert(QSIZE > numSlots, "QSIZE must be larger than numSlots");

 public:
  // pointer to shared data
  cuda::barrier<cuda::thread_scope_block> *barriers;
  T *data;
  // register data
  unsigned ptr;

  __device__ __forceinline__ uint64_t *native_bar(unsigned slot_id) {
    return cuda::device::barrier_native_handle(barriers[slot_id]);
  }

  __device__ __forceinline__ void wait() {
    barriers[ptr].arrive_and_wait();
  }

  template<int PH = 0>
  __device__ __forceinline__ T pop() {
    barriers[ptr].arrive_and_wait();
    T val = data[ptr];
    ptr = (ptr + 1) % QSIZE;
    return val;
  }

  template<int ThrPush = 0> 
  __device__ __forceinline__ void push(int tid, T val) {
    if (tid == ThrPush)
      data[ptr] = val;
    static_cast<void>(barriers[ptr].arrive());
    ptr = (ptr + 1) % QSIZE;
  }

  // single thread version of the push
  __device__ __forceinline__ void push(T val) {
    data[ptr] = val;
    static_cast<void>(barriers[ptr].arrive());
    ptr = (ptr + 1) % QSIZE;
  }

  __device__ __forceinline__ void put(T val) {
    data[ptr] = val;
  }
  __device__ __forceinline__ void advance() {
    ptr = (ptr + 1) % QSIZE;
  }
  __device__ __forceinline__ void commit() {
    static_cast<void>(barriers[ptr].arrive());
  }
};

static __device__ __forceinline__ uint16_t extract(const int s) {
  // TODO(zhiyuang): consider remove this -1
  return __ffs(s) - 1;
}

template<unsigned QSIZE = 32>
struct SizeBoundedBarrierAllocQueue : public SizeBoundedBarrierQueue<int, QSIZE> {
  using Base = SizeBoundedBarrierQueue<int, QSIZE>;

  uint32_t shared_avail;

  __device__ __forceinline__ SizeBoundedBarrierAllocQueue(
    cuda::barrier<cuda::thread_scope_block> *bars,
    int *datalist,
    uint16_t ptr,
    int *flaglist
  ) : Base{bars, datalist, ptr}, shared_avail(__cvta_generic_to_shared(flaglist)) {}

  __device__ __forceinline__ void reset(int msg) {
    // TODO(zhiyuang): replace the atomicOr with PTX atom.shared
    asm volatile(
      "red.shared.or.b32 [%0], %1;"
      :
      : "r"(shared_avail), "r"(msg)
      : "memory"
    );
    // atomicOr(avail, msg);
  }

  template<int ThrPush = 0, bool writeback = false, bool free_slot = true> 
  __device__ __forceinline__ void push(int tid, int val) {
    if constexpr (writeback) {
      if (val < 0)
        return;

      if (tid == ThrPush)
        this->data[this->ptr] = val;
      static_cast<void>(this->barriers[this->ptr].arrive());
      this->ptr = (this->ptr + 1) % QSIZE;
    } else if constexpr (free_slot) {
      // here, val >= rejects the val that have high bit set
      if (tid == ThrPush) {
        reset(val);
      }
    }
  }
};

// This is meant to be used as shared, and call from single thread only
template<typename T, size_t QSIZE = 32>
struct SingleThreadQueue {
  T data[QSIZE];
  int head, tail;

  __device__ __forceinline__ void init() {
      head = 0;
      tail = 0;
  }

  __device__ __forceinline__ bool empty() {
      return head == tail;
  }

  template<int Th>
  __device__ __forceinline__ void thread_push(T val) {
    int qslot = head % QSIZE;
    if (threadIdx.x == Th) {
        // printf("[SingleThreadQueue] thread_push: head=%d val=%d\n", head, val);
        data[qslot] = val;
        head++;
    }
  }

  __device__ __forceinline__ T warp_pop() {
    int qslot = tail % QSIZE;
    T val = data[qslot];
    if (cuda::ptx::get_sreg_laneid() == 0) {
        tail++;
    }
    __mprint("[SingleThreadQueue] warp_pop: tail=%d", tail);
    return val;
  }
};

template<typename T, unsigned QSIZE = 32>
struct BarrierQueue {
  // TODO(zhiyuang): replace this with parity barrier
  cuda::barrier<cuda::thread_scope_block> *write_barriers;
  cuda::barrier<cuda::thread_scope_block> *read_barriers;
  T *data;

  // register data
  unsigned head, tail;

  __device__ __forceinline__ void init(
      cuda::barrier<cuda::thread_scope_block> *_write_bars,
      cuda::barrier<cuda::thread_scope_block> *_read_bars,
      T *_data) {
    // threadIdx0 init
    if (threadIdx.x == 0) {
      write_barriers = _write_bars;
      read_barriers = _read_bars;
      data = _data;
    }

    // per-thread init
    head = 0;
    tail = 0;
  }

  __device__ __forceinline__ void pop_acquire() {
    // printf("[tid=%d][BarrierQueue] pop_acquire at slot=%d, barrier=%p\n", threadIdx.x, tail, &write_barriers[tail % QSIZE]);
    write_barriers[tail % QSIZE].arrive_and_wait();
  }
  
  __device__ __forceinline__ T pop_commit(int n = 1) {
    int qslot = tail % QSIZE;
    T val = data[qslot];
    static_cast<void>(read_barriers[qslot].arrive(n));
    tail++;
    return val;
  }
  __device__ __forceinline__ T pop(int n = 1) {
    pop_acquire();
    return pop_commit(n);
  }

  __device__ __forceinline__ uint8_t push_acquire() {
    // just wait!
    int qslot = head % QSIZE;
    // __mprint("[BarrierQueue] push_acquire at slot head=%d qslot=%d", head, qslot);

    read_barriers[qslot].arrive_and_wait();
    return qslot;
  }

  template <int Th>
  __device__ __forceinline__ void push_commit(T val) {
    // __mprint("[BarrierQueue] push_commit at slot %d val=%d", head, val);
    if (__memory_tid() == Th)
        data[head % QSIZE] = val;
    int qslot = head % QSIZE;
    // printf("[BarrierQueue] push_commit arrive at slot head=%d qslot=%d", head, qslot);
    static_cast<void>(write_barriers[qslot].arrive());
    head ++;
  }

  template<int Th> 
  __device__ __forceinline__ void push(T val) {
    (void)push_acquire();
    push_commit<Th>(val);
  }
};

template<typename T, unsigned QSIZE = 32>
struct SingleBarrierQueue {
  // TODO(zhiyuang): replace this with parity barrier
  cuda::barrier<cuda::thread_scope_block> *write_barriers;
  T *data;
  unsigned *shared_tail;

  // register data
  unsigned head, tail;

  __device__ __forceinline__ void init(
      cuda::barrier<cuda::thread_scope_block> *_write_bars,
      unsigned *_shared_tail,
      T *_data) {
    // per-thread init
    write_barriers = _write_bars;
    data = _data;
    shared_tail = _shared_tail;

    head = tail = 0;
  }

  template <int Th>
  __device__ __forceinline__ T pop(int n = 1) {
    unsigned qslot = tail % QSIZE;
    write_barriers[qslot].arrive_and_wait();
    T val = data[qslot];
    tail ++;
    if (threadIdx.x == Th)
      *shared_tail = tail;
    return val;
  }

  __device__ __forceinline__ uint8_t push_acquire() {
    // TODO(zhiyuang): spin wait here. This resource is critical
    // likely?
    while (head - tail >= QSIZE) [[unlikely]] {
      tail = *shared_tail;
    }
    return head % QSIZE;
  }

  template <int Th>
  __device__ __forceinline__ void push_commit(T val, unsigned qslot) {
    // __mprint("[BarrierQueue] push_commit at slot %d val=%d", qslot, val);
    if (__memory_tid() == Th) {
        data[qslot] = val;
    }
    // printf("[BarrierQueue] push_commit arrive at slot head=%d qslot=%d", head, qslot);
    static_cast<void>(write_barriers[qslot].arrive());
    head++;
  }

  template<int Th> 
  __device__ __forceinline__ void push(T val) {
    unsigned qslot = push_acquire();
    push_commit<Th>(val, qslot);
  }
};