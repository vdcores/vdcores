// name, wire opcode, compile-time execute-warp type
//
// A PoolInst is not interpreted by a generic pool VM.  Each entry binds its
// wire opcode to the concrete multi-warp executor instantiated by dae2.  Host
// launch dispatch selects one of these precompiled assemblies; device code
// never switches on PoolInst::opcode.
DAE_POOL_OP(POOL_SLICE_EXCHANGE, 1, PoolSliceExchangeExecuteWarp)
