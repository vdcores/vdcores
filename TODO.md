## Demo TODOs

- [x] embedding, which will copy from embedding table to hidden
  - [x] in decoding, fuse(argfuse) with rmsnorm
  - [x] * copy (sched to sm128+) to matHidden
- [x] argmax
- [x] RMS norm: load the parameter from the tensor instead of gmem
  - [ ] use tensor1d to load the matrix, see comments in the new file
- [x] attention layout: the input and output O
  - QWen-8B uses QK norm
  - input should just be [N_REQ, HIDDEN], remove the seqlen field (or set to 1?)
  - output should just be [N_REQ, HIDDEN], use tma1d store here (or set to 1?)
- [ ] loop-based operations
  - [ ] better barrier compared to issue barrier
- [ ] placement, esp reduce the poll on global memory
- [ ] copy Q for multitoken mode

Verification
- [x] * V do not need ROPE
- [ ] Correctness of multi-layer, compare with transformers?

BugFix:
- [ ] raw address allocation
- [ ] unify the cord converter

Optmizations
- [ ] optimize layout for KV; make them stay same in the single batch?
  - [ ] how do paged attention do?
- [ ] direct bar from raw address (spcifically, argmax)
- [ ] L2 cache policy
- [ ] ldmatrix and stmatrix - copy atom for writing back frags to smem?
- [ ] single-layer optimization
  - [ ] optimize raw address, make it use the other bar (more generally, make others also)
  - [ ] preload barrier (esp, immidiate resolved ones - add fast load & preload path?)
    - could be a mix of initial L1 load and later L2 based loads
  - [ ] GEMV kernel optimization
  - [ ] multiple dispatch queues
  - [ ] reorder the compute instructions
- [ ] Runtime optimization
  - [ ] prefetch of instructions w/ global optimizations
  - [ ] change the jump flag - remove and replace with a repeat count
- [ ] limit the inflights by bytes
  - [ ] implementation using the arrive (reflect on the compute side)
