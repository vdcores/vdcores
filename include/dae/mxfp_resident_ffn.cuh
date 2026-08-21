#pragma once

// Shared-memory contracts for coupled MX streams. The resident MXFP4/MXFP8
// FFN kinds use fixed addresses; the common MXFP8 projection kind below uses
// an ordinary allocator-owned retained-ring lease.
namespace dae_mxfp_resident_ffn {

enum CoupledStreamKind : uint16_t {
  kCoupledLinear1 = 0,
  kCoupledDownWeight = 1,
  kCoupledDownActivation = 2,
  // Allocator-owned, projection-agnostic MXFP8 x MXFP8 GEMV stream.  Unlike
  // the three resident-FFN kinds above, this form publishes one normal M2C
  // lease and sends the same immutable command to both LDUs.
  kCoupledFp8Gemv = 3,
  // Descriptor-driven allocator-owned internal ring.  The device plan
  // describes rank, coordinates, iteration/issue strides, and one lane per
  // LDU port.  This is intentionally operand-agnostic; attention is its first
  // consumer, not part of the operator contract.
  kCoupledTmaRing = 4,
};

static constexpr uint16_t kCoupledKindMask = 0x000f;
static constexpr int kCoupledStagesShift = 4;
static constexpr uint16_t kCoupledStagesMask = 0x00f0;
static constexpr uint16_t kCoupledLocalChain = 0x0100;
// The fixed-area resident command resolves its expert task base from the
// prepared 128-byte route record before issuing either weight or scale TMA.
static constexpr uint16_t kCoupledDynamicExpert = 0x0200;
static constexpr int kCoupledPhaseBaseShift = 9;
static constexpr uint16_t kCoupledPhaseBaseMask = 0xfe00;
static constexpr int kCoupledPortMaskShift = 9;
static constexpr uint16_t kCoupledPortMask = 0x0600;
// The common plan is two uint64 pointers. A size-field flag asks the allocator
// to select the current resident layer's 16-byte plan record before publishing
// the otherwise unchanged command to both LDUs.
static constexpr uint16_t kCoupledLayerIndexedSize = 0x8000;
static constexpr uint16_t kCoupledStreamLengthMask = 0x7fff;

// Common native MXFP8 x MXFP8 ring. Each retained K256 stage carries both
// M128 output groups, their packed SFA images, and the shared activation/SFB.
static constexpr int kFp8CoupledStages = 2;
static constexpr int kFp8CoupledWeightDataBytes = 4 * 128 * 128;
static constexpr int kFp8CoupledWeightScaleBytes = 2 * 512;
static constexpr int kFp8CoupledActivationDataBytes = 2 * 8 * 128;
static constexpr int kFp8CoupledActivationScaleBytes = 1024;
static constexpr int kFp8CoupledWeightScaleOffset =
    kFp8CoupledWeightDataBytes;
static constexpr int kFp8CoupledActivationDataOffset =
    kFp8CoupledWeightScaleOffset + kFp8CoupledWeightScaleBytes;
static constexpr int kFp8CoupledActivationScaleOffset =
    kFp8CoupledActivationDataOffset + kFp8CoupledActivationDataBytes;
static constexpr int kFp8CoupledStageBytes = 68 * 1024;
static constexpr int kFp8CoupledRingBytes =
    kFp8CoupledStages * kFp8CoupledStageBytes;
static constexpr int kFp8CoupledAreaSlots =
    (kFp8CoupledRingBytes + 8 * 1024 - 1) / (8 * 1024);
static_assert(
    kFp8CoupledActivationScaleOffset +
        kFp8CoupledActivationScaleBytes <= kFp8CoupledStageBytes);
static_assert(kFp8CoupledRingBytes == 136 * 1024);
static_assert(kFp8CoupledAreaSlots == 17);

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
static constexpr int kLinear1AreaSlots =
    kLinear1OperandBytes / (8 * 1024);

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
static constexpr int kDownAreaSlots =
    (kDownOperandBytes + 8 * 1024 - 1) / (8 * 1024);

static_assert(kLinear1WeightRingOffset % 1024 == 0);
static_assert(kLinear1ScaleRingOffset % 1024 == 0);
static_assert(kLinear1ActivationOffset % 1024 == 0);
static_assert(kDownScaleRingOffset % 1024 == 0);
static_assert(kDownActivationRingOffset % 1024 == 0);
static_assert(kDownOutputOffset % 1024 == 0);
static_assert(kDownOperandBytes <= kLinear1ScaleRingOffset);
static_assert(kLinear1OperandBytes % (8 * 1024) == 0);

}  // namespace dae_mxfp_resident_ffn
