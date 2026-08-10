#include <cute/arch/mma_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/numeric_types.h>

#include <cstdio>

int main() {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, float, cutlass::float_ue4m3_t,
      128, 8, 16, UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));

  TiledMma tiled_mma;
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<128>{}, Int<256>{}));
  auto mma_layout_a = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp4>{}, mma_shape_a);
  auto logical_layout_a = tile_to_shape(
      UMMA::Layout_K_SW128_Atom<Fp4>{},
      make_shape(Int<128>{}, Int<256>{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
  using TileShape = Shape<Int<128>, Int<8>, Int<256>>;
  auto shared_sfa = ScaleConfig::deduce_smem_layoutSFA(
      tiled_mma, TileShape{});
  auto shared_sfb = ScaleConfig::deduce_smem_layoutSFB(
      tiled_mma, TileShape{});
  auto global_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      Shape<Int<128>, Int<128>, Int<256>>{});
  auto global_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      Shape<Int<128>, Int<128>, Int<256>>{});

  std::printf("mma_shape_a: ");
  print(mma_shape_a);
  std::printf("\nmma_layout_a: ");
  print(mma_layout_a);
  std::printf("\nlogical_layout_a: ");
  print(logical_layout_a);
  std::printf("\nshared_sfa: ");
  print(shared_sfa);
  std::printf("\nshared_sfb: ");
  print(shared_sfb);
  std::printf("\nglobal_sfa: ");
  print(global_sfa);
  std::printf("\nglobal_sfb: ");
  print(global_sfb);
  std::printf("\n");
  for (int row = 0; row < 8; ++row) {
    std::printf("row=%d", row);
    for (int k = 0; k < 256; k += 16) {
      int mma_offset = mma_layout_a(make_coord(
          make_coord(row, k % TiledMma::K), 0, k / TiledMma::K));
      int logical_offset = logical_layout_a(row, k);
      std::printf(" k%d=%d/%d", k, mma_offset, logical_offset);
    }
    std::printf("\n");
  }
  int scale_mismatches = 0;
  for (int row = 0; row < 128; ++row) {
    for (int sf = 0; sf < 16; ++sf) {
      auto mma_coord = make_coord(
          make_coord(make_coord(row % 32, row / 32), 0),
          make_coord(0, sf % 4));
      auto shared_coord = make_coord(
          mma_coord, 0, make_coord(0, sf / 4));
      int shared_offset = shared_sfa(shared_coord);
      int global_offset = global_sfa(row, sf * 16);
      if (shared_offset != global_offset) {
        if (scale_mismatches < 32) {
          std::printf(
              "scale mismatch row=%d sf=%d shared=%d global=%d\n",
              row, sf, shared_offset, global_offset);
        }
        ++scale_mismatches;
      }
    }
  }
  std::printf("scale_mismatches=%d\n", scale_mismatches);
  for (int row = 0; row < 8; ++row) {
    std::printf("scale row=%d", row);
    for (int sf = 0; sf < 16; ++sf) {
      std::printf(" sf%d=%d", sf, int(global_sfa(row, sf * 16)));
    }
    std::printf("\n");
  }
  int nibble_pair_mismatches = 0;
  for (int row = 0; row < 8; ++row) {
    for (int k = 0; k < 256; k += 2) {
      int lo = mma_layout_a(make_coord(
          make_coord(row, k % TiledMma::K), 0, k / TiledMma::K));
      int hi = mma_layout_a(make_coord(
          make_coord(row, (k + 1) % TiledMma::K), 0,
          (k + 1) / TiledMma::K));
      if (hi != lo + 1 || (lo & 1) != 0) {
        if (nibble_pair_mismatches < 32) {
          std::printf(
              "nibble mismatch row=%d k=%d lo=%d hi=%d\n",
              row, k, lo, hi);
        }
        ++nibble_pair_mismatches;
      }
    }
  }
  std::printf("nibble_pair_mismatches=%d\n", nibble_pair_mismatches);
}
