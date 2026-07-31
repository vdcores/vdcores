// name, wire opcode, compile-time execute-warp type
//
// A PoolInst is not interpreted by a generic pool VM.  Each entry binds its
// wire opcode to the concrete multi-warp executor instantiated by dae2.  Host
// launch dispatch selects one of these precompiled assemblies; device code
// never switches on PoolInst::opcode.
#if defined(DAE_ENABLE_NVSHMEM) || defined(DAE_ENABLE_LOCAL_POOL)
DAE_POOL_OP(POOL_SLICE_EXCHANGE, 1, PoolSliceExchangeExecuteWarp)
DAE_POOL_OP(
    POOL_SLICE_WEIGHTED_EXCHANGE,
    2,
    PoolSliceWeightedExchangeExecuteWarp)
DAE_POOL_OP(
    POOL_SLICE_HOST_WEIGHTED_EXCHANGE,
    3,
    PoolSliceHostWeightedExchangeExecuteWarp)
#ifdef DAE_ENABLE_LOCAL_POOL
DAE_POOL_OP(
    POOL_SLICE_MULTIMEM_EXCHANGE,
    5,
    PoolSliceMultimemExchangeExecuteWarp)
#endif
#endif
#ifdef DAE_ENABLE_NCCL_GIN
DAE_POOL_OP(
    POOL_SLICE_GIN_WEIGHTED_EXCHANGE,
    4,
    PoolSliceGinWeightedExchangeExecuteWarp)
#endif
