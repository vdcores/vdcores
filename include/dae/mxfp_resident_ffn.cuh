#pragma once

// Fixed shared-memory contract for the dedicated resident MXFP4/MXFP8 FFN
// path.  The memory and compute virtual cores use these addresses directly;
// no allocator lease or ring-base publication is part of the protocol.
namespace dae_mxfp_resident_ffn {

static constexpr int kLinear1Stages = 2;
static constexpr int kLinear1WeightStageBytes = 64 * 1024;
static constexpr int kLinear1WeightRingOffset = 0;
static constexpr int kLinear1WeightRingBytes =
    kLinear1Stages * kLinear1WeightStageBytes;

static constexpr int kLinear1ScaleStageBytes = 4 * 1024;
static constexpr int kLinear1ScaleRingOffset =
    kLinear1WeightRingOffset + kLinear1WeightRingBytes;
static constexpr int kLinear1ScaleRingBytes =
    kLinear1Stages * kLinear1ScaleStageBytes;

static constexpr int kLinear1ActivationOffset =
    kLinear1ScaleRingOffset + kLinear1ScaleRingBytes;
static constexpr int kLinear1ActivationBytes = 32 * 1024;
static constexpr int kLinear1OperandBytes =
    kLinear1ActivationOffset + kLinear1ActivationBytes;

// Linear-2 immediately reuses the dead Linear-1 operand arena. Weight/SFA
// production is independent of Linear-1 output readiness; activation data
// and SFB have a separate full barrier after their per-expert dependency.
static constexpr int kDownStages = 2;
static constexpr int kDownWeightStageBytes = 32 * 1024;
static constexpr int kDownWeightRingOffset = 0;
static constexpr int kDownWeightRingBytes =
    kDownStages * kDownWeightStageBytes;
static constexpr int kDownScaleStageBytes = 2 * 1024;
static constexpr int kDownScaleRingOffset =
    kDownWeightRingOffset + kDownWeightRingBytes;
static constexpr int kDownScaleRingBytes =
    kDownStages * kDownScaleStageBytes;
static constexpr int kDownActivationStageBytes = 2 * 1024;
static constexpr int kDownActivationRingOffset =
    kDownScaleRingOffset + kDownScaleRingBytes;
static constexpr int kDownActivationRingBytes =
    kDownStages * kDownActivationStageBytes;
static constexpr int kDownOutputOffset =
    kDownActivationRingOffset + kDownActivationRingBytes;
static constexpr int kDownOutputBytes = 4 * 1024;
static constexpr int kDownOperandBytes = kDownOutputOffset + kDownOutputBytes;

static_assert(kLinear1WeightRingOffset % 1024 == 0);
static_assert(kLinear1ScaleRingOffset % 1024 == 0);
static_assert(kLinear1ActivationOffset % 1024 == 0);
static_assert(kDownScaleRingOffset % 1024 == 0);
static_assert(kDownActivationRingOffset % 1024 == 0);
static_assert(kDownOutputOffset % 1024 == 0);
static_assert(kDownOperandBytes <= kLinear1ScaleRingOffset);

}  // namespace dae_mxfp_resident_ffn
