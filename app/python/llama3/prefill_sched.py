import argparse
import math
import os
import time
from functools import partial

import torch
import torch.nn.functional as F
from dae.instructions import (
    Gemm_M64N64,
    Gemm_M64N64K64,
    Gemm_M64N128K64,
    Gemv_M64N8,
    ROPE_INTERLEAVE_512,
    RegLoad,
    RegStore,
    SILU_MUL_SHARED_BF16_K_64_SW128,
    TmaLoad1D,
    TmaStore1D,
)
from dae.launcher import Launcher
from dae.model import (
    build_tma_wgmma_k,
    build_tma_wgmma_mn,
    cord_func_K_major,
    cord_func_MN_major,
    cord_gqa_load_q,
    tma_gqa_load_q,
)
from dae.schedule import (
    IssueBarrier,
    ListSchedule,
    LoopC,
    LoopM,
    RepeatM,
    SchedAttentionDecoding,
    SchedCopy,
    SchedGemm,
    SchedGemv,
    SchedRMSShared,
    SchedRegSiLUFused,
    SchedRope,
    SchedSmemSiLUInterleaved,
    Schedule,
)
from dae.tma_utils import Major, ToRopeTableCordAdapter, ToSplitMCordAdapter, cord_load_tbl, tma_load_tbl
from dae.util import DEFAULT_COMPUTE_OPS_FILE, dump_insts, write_compute_operator_file
from reference import check_tensor_threshold, reference_pass
from transformers import AutoConfig, AutoModelForCausalLM


class SchedAttentionTokenList(Schedule):
    def __init__(self, seq_lens, kv_block_size, num_kv_heads, matO, tmas):
        super().__init__()
        self.seq_lens = list(seq_lens)
        self.kv_block_size = kv_block_size
        self.num_heads = num_kv_heads
        self.matO = matO
        self.tmas = tmas
        self.required_sms = len(self.seq_lens) * self.num_heads
        self.AttentionInst = SchedAttentionDecoding(reqs=1, seq_len=1, KV_BLOCK_SIZE=kv_block_size, NUM_KV_HEADS=num_kv_heads, matO=matO[:1], tmas=tmas).AttentionInst

    def _on_place(self):
        assert 0 < self.num_sms <= self.required_sms, (
            f"SchedAttentionTokenList requires 0 < sms <= {self.required_sms}, got {self.num_sms}"
        )

    def schedule(self, sm: int):
        if sm < 0:
            return []

        tQ, tK, tV = self.tmas
        insts = []
        for job in range(sm, self.required_sms, self.num_sms):
            token_idx = job // self.num_heads
            head = job % self.num_heads
            seq_len = self.seq_lens[token_idx]
            num_kv_blocks = (seq_len + self.kv_block_size - 1) // self.kv_block_size
            seq_len_last_block = seq_len % self.kv_block_size
            if seq_len_last_block == 0:
                seq_len_last_block = self.kv_block_size

            insts.extend(
                [
                    self.AttentionInst(num_kv_blocks, seq_len_last_block, need_norm=False, need_rope=False),
                    tQ.cord(token_idx, head).bar(self._bar("q")).group(),
                ]
            )
            if num_kv_blocks > 1:
                insts.append(
                    RepeatM.on(
                        num_kv_blocks - 1,
                        [tK.cord(0, 0, head, 0).group(), tK.cord2tma(0, self.kv_block_size, 0, 0)],
                        [tV.cord(0, 0, head, 0).group(), tV.cord2tma(0, self.kv_block_size, 0, 0)],
                    )
                )
            insts.extend(
                [
                    tK.cord(0, self.kv_block_size * (num_kv_blocks - 1), head, 0).bar(self._bar("k")).group(),
                    tV.cord(0, self.kv_block_size * (num_kv_blocks - 1), head, 0).group(),
                    TmaStore1D(self.matO[token_idx, head], numSlots=2).bar(self._bar("o")).group(),
                ]
            )
        return insts

    def bar_release_count(self, role: str):
        if role != "o":
            return 0
        return self._bar_release_if_present(role, self.required_sms)


class ToPrefillKVStoreCordAdapter:
    def __init__(self, inner, token_base: int, tile_m: int, num_m_tiles: int | None = None):
        self.inner = inner
        self.token_base = token_base
        self.tile_m = tile_m
        self.num_m_tiles = num_m_tiles

    def cord(self, *cords):
        if len(cords) == 1:
            sm = cords[0]
            if self.num_m_tiles is not None:
                sm %= self.num_m_tiles
            m = sm * self.tile_m
        elif len(cords) == 2:
            _, m = cords
        else:
            raise ValueError(f"unexpected cords for prefill KV store: {cords}")
        return self.inner.cord(self.token_base, m)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ToTokenBlockNCordAdapter:
    def __init__(self, inner, token_base: int):
        self.inner = inner
        self.token_base = token_base

    def cord(self, *cords):
        if len(cords) != 2:
            raise ValueError(f"unexpected cords for token block adapter: {cords}")
        n, other = cords
        return self.inner.cord(self.token_base + n, other)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SchedSmemSiLUSw128(Schedule):
    def __init__(self, num_token: int, num_tiles: int, tmas, base_offset: int = 0):
        super().__init__()
        self.num_token = num_token
        self.num_tiles = num_tiles
        self.load_gate, self.load_up, self.store_out = tmas
        self.base_offset = base_offset

    def _on_place(self):
        assert self.num_sms == self.num_tiles, f"SchedSmemSiLUSw128 expects {self.num_tiles} SMS, got {self.num_sms}"

    def schedule(self, sm: int):
        if sm < 0:
            return []
        col = self.base_offset + sm * 64
        return [
            SILU_MUL_SHARED_BF16_K_64_SW128(self.num_token),
            self.store_out.cord(0, col).bar(self._bar("output")).group(),
            self.load_gate.cord(0, col).bar(self._bar("input")).group(),
            self.load_up.cord(0, col),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_tiles)


def permute_rope_weight(weight, head_dim, num_heads):
    return weight.view(num_heads, 2, head_dim // 2, weight.shape[-1]).transpose(1, 2).reshape_as(weight).contiguous()


def permute_rope_rows(rows, head_dim, num_heads):
    return rows.view(rows.shape[0], num_heads, 2, head_dim // 2).transpose(2, 3).reshape_as(rows).contiguous()


def build_inputs(token_ids: torch.Tensor):
    positions = torch.arange(token_ids.numel(), dtype=torch.long, device=token_ids.device).unsqueeze(0)
    return {
        "input_ids": token_ids.unsqueeze(0),
        "attention_mask": torch.ones((1, token_ids.numel()), dtype=torch.long, device=token_ids.device),
        "position_ids": positions,
    }


def manual_attention_chunk(q_chunk, k_full, v_full, head_group_size, prefix_len, chunk, head_dim):
    q = q_chunk.view(chunk, -1, head_group_size, head_dim).permute(1, 2, 0, 3).contiguous()
    k = k_full.view(prefix_len + chunk, -1, head_dim).permute(1, 0, 2).contiguous()
    v = v_full.view(prefix_len + chunk, -1, head_dim).permute(1, 0, 2).contiguous()

    q_pos = torch.arange(prefix_len, prefix_len + chunk, device=q_chunk.device)
    k_pos = torch.arange(prefix_len + chunk, device=q_chunk.device)
    mask = (k_pos.unsqueeze(0) > q_pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(q.float(), k.float().unsqueeze(1).transpose(-1, -2)) / math.sqrt(head_dim)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float().unsqueeze(1))
    return out.permute(2, 0, 1, 3).reshape(chunk, -1).to(torch.bfloat16)


def apply_interleaved_rope(hidden_states: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    cos = rope[..., 0::2].float()
    sin = rope[..., 1::2].float()
    even = hidden_states_f32[..., 0::2]
    odd = hidden_states_f32[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)
    return rotated.to(hidden_states.dtype)


def main():
    parser = argparse.ArgumentParser(description="Llama3-8B prefill chunk schedule")
    parser.add_argument("--hf-cache-dir", default="/tmp/huggingface_cache")
    parser.add_argument("--prefill-len", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--chunk-start", type=int, default=None)
    parser.add_argument("--debug-num-layers", type=int, default=None)
    parser.add_argument("--debug-stop-after", choices=["attention", "out_proj", "post_attn_rms", "silu", "down_proj"], default=None)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--bench", type=int, default=0)
    parser.add_argument("--instdump", type=int, default=None)
    parser.add_argument("-w", "--write-compute-ops", type=str, nargs="?", const=DEFAULT_COMPUTE_OPS_FILE, default=None)
    args = parser.parse_args()

    if args.chunk_size not in (8, 16, 32, 64, 128, 256):
        raise ValueError("This schedule currently supports --chunk-size in {8,16,32,64,128,256}")
    if args.chunk_size % 8 != 0:
        raise ValueError("chunk_size must be a multiple of 8")
    if args.debug_stop_after is not None and args.debug_num_layers not in (None, 1):
        raise ValueError("debug stop mode currently supports exactly one layer")
    if args.chunk_start is None:
        args.chunk_start = args.prefill_len - args.chunk_size
    if args.chunk_start < 0 or args.chunk_start + args.chunk_size > args.prefill_len:
        raise ValueError("chunk window must lie inside the prefill length")
    if args.chunk_start % args.chunk_size != 0:
        raise ValueError("chunk_start must be aligned to chunk_size")
    if args.prefill_len >= 1024:
        raise ValueError("Use a mid-length sequence under 1K for this harness")
    if not (args.correctness or args.launch or args.bench or args.instdump is not None or args.write_compute_ops is not None):
        parser.error("select at least one of --correctness, --launch, --bench, --instdump, or --write-compute-ops")

    torch.manual_seed(0)
    gpu = torch.device("cuda")

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=args.hf_cache_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=args.hf_cache_dir,
        token=os.environ["HF_TOKEN"],
    )

    eps = config.rms_norm_eps
    layers = model.model.layers
    if args.debug_num_layers is not None:
        layers = layers[:args.debug_num_layers]
    dtype = model.dtype
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = hidden // config.num_attention_heads
    qw = hidden
    kw = head_dim * config.num_key_value_heads
    vw = kw
    num_layers = len(layers)
    num_kv_heads = config.num_key_value_heads
    head_group_size = config.num_attention_heads // config.num_key_value_heads
    kv_block_size = 64
    chunk = args.chunk_size
    token_slots = chunk if (chunk == 8 or chunk >= 64) else 64
    num_q_blocks = chunk // 8
    prefix_len = args.chunk_start
    stop_after_final_kvq = False

    seq_token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.prefill_len,),
        generator=torch.Generator(device="cpu").manual_seed(0),
        dtype=torch.long,
    ).to(gpu)
    chunk_token_ids = seq_token_ids[prefix_len:prefix_len + chunk]
    chunk_positions = torch.arange(prefix_len, prefix_len + chunk, dtype=torch.long, device=gpu)

    attention_only_refs = None
    if args.debug_stop_after == "attention":
        with torch.no_grad():
            captured_attn, _ = reference_pass(model, build_inputs(seq_token_ids))
        attention_only_refs = {
            "q_chunk": permute_rope_rows(
                captured_attn[0]["q_proj"][0, prefix_len:prefix_len + chunk].contiguous(),
                head_dim,
                config.num_attention_heads,
            ),
            "k_full": permute_rope_rows(
                captured_attn[0]["k_proj"][0, :prefix_len + chunk].contiguous(),
                head_dim,
                config.num_key_value_heads,
            ),
            "v_full": captured_attn[0]["v_proj"][0, :prefix_len + chunk].contiguous(),
        }

    with torch.no_grad():
        if prefix_len > 0:
            prefix_inputs = build_inputs(seq_token_ids[:prefix_len])
            prefix_outputs = model(**prefix_inputs, use_cache=True)
            prefix_cache = prefix_outputs.past_key_values
        else:
            prefix_cache = []

    dae = Launcher(132, device=gpu)
    defaultg = dae.get_group()
    layerg = dae.add_group("layer", num_layers)

    layerg.addBarrier("bar_layer")
    layerg.addBarrier("bar_pre_attn_rms")
    layerg.addBarrier("bar_q_proj")
    layerg.addBarrier("bar_qkv_attn")
    layerg.addBarrier("bar_attn_out")
    layerg.addBarrier("bar_out_mlp")
    layerg.addBarrier("bar_post_attn_rms")
    layerg.addBarrier("bar_silu_in")
    layerg.addBarrier("bar_silu_out1")
    layerg.addBarrier("bar_silu_out2")

    matRope = torch.zeros(num_q_blocks, 8, head_dim, dtype=torch.bfloat16, device=gpu)
    rope_rows = torch.zeros(chunk, head_dim, dtype=torch.bfloat16, device=gpu)
    pos_cache = torch.arange(args.prefill_len, device=gpu).unsqueeze(0)
    cos_cache, sin_cache = model.model.rotary_emb(torch.zeros(1, device=gpu), pos_cache)
    for idx, pos in enumerate(chunk_positions.tolist()):
        block_idx = idx // 8
        block_token = idx % 8
        matRope[block_idx, block_token, 0::2] = cos_cache[0, pos, : head_dim // 2]
        matRope[block_idx, block_token, 1::2] = sin_cache[0, pos, : head_dim // 2]
        rope_rows[idx, 0::2] = cos_cache[0, pos, : head_dim // 2]
        rope_rows[idx, 1::2] = sin_cache[0, pos, : head_dim // 2]

    matHidden = torch.zeros(token_slots, hidden, dtype=dtype, device=gpu)
    matRMSHidden = torch.zeros_like(matHidden)
    attnQs = [torch.zeros(token_slots, qw, dtype=dtype, device=gpu) for _ in range(num_layers)]
    attnKs = [torch.zeros(args.prefill_len, kw, dtype=dtype, device=gpu) for _ in range(num_layers)]
    attnVs = [torch.zeros(args.prefill_len, vw, dtype=dtype, device=gpu) for _ in range(num_layers)]
    attnO = torch.zeros(token_slots, hidden, dtype=dtype, device=gpu)
    matInterm = torch.zeros(token_slots, intermediate, dtype=dtype, device=gpu)
    matGateOut = torch.zeros(token_slots, intermediate, dtype=dtype, device=gpu)
    matSiLUOut = torch.zeros(token_slots, intermediate, dtype=dtype, device=gpu)

    with torch.no_grad():
        cache_layers = getattr(prefix_cache, "layers", prefix_cache)
        for layer_idx, layer_cache in enumerate(cache_layers[:num_layers]):
            if hasattr(layer_cache, "keys"):
                keys = layer_cache.keys
                values = layer_cache.values
            else:
                keys, values = layer_cache[0], layer_cache[1]
            k_cache = keys[0].permute(1, 0, 2).reshape(prefix_len, kw)
            v_cache = values[0].permute(1, 0, 2).reshape(prefix_len, vw)
            attnKs[layer_idx][:prefix_len].copy_(permute_rope_rows(k_cache, head_dim, num_kv_heads))
            attnVs[layer_idx][:prefix_len].copy_(v_cache)

    matRMSInputW = [layer.input_layernorm.weight for layer in layers] + [model.model.norm.weight]
    matRMSPostAttnW = [layer.post_attention_layernorm.weight for layer in layers]
    matqWs = [permute_rope_weight(layer.self_attn.q_proj.weight, head_dim, config.num_attention_heads) for layer in layers]
    matkWs = [permute_rope_weight(layer.self_attn.k_proj.weight, head_dim, config.num_key_value_heads) for layer in layers]
    matvWs = [layer.self_attn.v_proj.weight for layer in layers]
    matOutWs = [layer.self_attn.o_proj.weight for layer in layers]
    matUps = [layer.mlp.up_proj.weight for layer in layers]
    matGates = [layer.mlp.gate_proj.weight for layer in layers]
    matDowns = [layer.mlp.down_proj.weight for layer in layers]

    dae.set_streaming(matqWs, matkWs, matvWs, matOutWs, matUps, matGates, matDowns)

    defaultg.addTma("loadRope", [matRope], lambda t: t._build("load", 64, 8, tma_load_tbl, cord_load_tbl))

    gemv_tile_m, _, gemv_tile_k = Gemv_M64N8.MNK
    gemv_input_tile = gemv_tile_k * Gemv_M64N8.n_batch
    wide_tile_m, wide_tile_n, wide_tile_k = Gemm_M64N64K64.MNK
    widek128_tile_m, widek128_tile_n, widek128_tile_k = Gemm_M64N64.MNK
    wide128_tile_m, wide128_tile_n, wide128_tile_k = Gemm_M64N128K64.MNK

    layerg.addTma("loadRMSBlock", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(8, gemv_input_tile, Major.K))
    layerg.addTma("storeQBlock", attnQs, lambda t: t.wgmma("reduce", 8, 64, Major.MN))
    layerg.addTma("storeKCacheBlock", attnKs, lambda t: t.wgmma("reduce", 8, 64, Major.MN))
    layerg.addTma("storeVCacheBlock", attnVs, lambda t: t.wgmma("reduce", 8, 64, Major.MN))

    if token_slots > 8:
        layerg.addTma("loadRMSWide", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(wide_tile_m, wide_tile_k, Major.K))
        layerg.addTma("loadRMSWideK128", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(widek128_tile_m, widek128_tile_k, Major.K))
        layerg.addTma("loadAttnOWide", [attnO] * num_layers, lambda t: t.wgmma_load(wide_tile_m, wide_tile_k, Major.K))
        layerg.addTma("loadSiluWide", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(wide_tile_m, wide_tile_k, Major.K))
        layerg.addTma("reduceHiddenWide", [matHidden] * num_layers, lambda t: t.wgmma("reduce", wide_tile_m, wide_tile_n, Major.K))
        layerg.addTma("storeIntermWide", [matInterm] * num_layers, lambda t: t.wgmma_store(wide_tile_m, wide_tile_n, Major.K))
        layerg.addTma("storeGateWide", [matGateOut] * num_layers, lambda t: t.wgmma_store(wide_tile_m, wide_tile_n, Major.K))
        layerg.addTma("loadRMSWide128", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(wide128_tile_m, wide128_tile_k, Major.K))
        layerg.addTma("loadAttnOWide128", [attnO] * num_layers, lambda t: t.wgmma_load(wide128_tile_m, wide128_tile_k, Major.K))
        layerg.addTma("loadSiluWide128", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(wide128_tile_m, wide128_tile_k, Major.K))
        layerg.addTma("reduceHiddenWide128", [matHidden] * num_layers, lambda t: t.wgmma("reduce", wide128_tile_m, wide128_tile_n, Major.K))
        layerg.addTma("storeIntermWide128", [matInterm] * num_layers, lambda t: t.wgmma_store(wide128_tile_m, wide128_tile_n, Major.K))
        layerg.addTma("storeGateWide128", [matGateOut] * num_layers, lambda t: t.wgmma_store(wide128_tile_m, wide128_tile_n, Major.K))
        layerg.addTma("storeVCacheWide", attnVs, lambda t: t.wgmma_store(wide_tile_m, wide_tile_n, Major.K))
        layerg.addTma("loadOutWide", matOutWs, lambda t: t.wgmma_load(wide_tile_n, wide_tile_k, Major.K))
        layerg.addTma("loadDownWide", matDowns, lambda t: t.wgmma_load(wide_tile_n, wide_tile_k, Major.K))
        layerg.addTma("loadUpWide", matUps, lambda t: t.wgmma_load(wide_tile_n, wide_tile_k, Major.K))
        layerg.addTma("loadGateWide", matGates, lambda t: t.wgmma_load(wide_tile_n, wide_tile_k, Major.K))
        layerg.addTma("loadGateWideK128", matGates, lambda t: t.wgmma_load(widek128_tile_n, widek128_tile_k, Major.K))
        layerg.addTma("loadOutWide128", matOutWs, lambda t: t.wgmma_load(wide128_tile_n, wide128_tile_k, Major.K))
        layerg.addTma("loadDownWide128", matDowns, lambda t: t.wgmma_load(wide128_tile_n, wide128_tile_k, Major.K))
        layerg.addTma("loadUpWide128", matUps, lambda t: t.wgmma_load(wide128_tile_n, wide128_tile_k, Major.K))
        layerg.addTma("loadGateWide128", matGates, lambda t: t.wgmma_load(wide128_tile_n, wide128_tile_k, Major.K))
        layerg.addTma("loadVWide", matvWs, lambda t: t.wgmma_load(wide_tile_n, wide_tile_k, Major.K))
        if chunk < 64:
            layerg.addTma("loadRMSChunk", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
            layerg.addTma("loadAttnOChunk", [attnO] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
            layerg.addTma("loadSiluChunk", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
            layerg.addTma("reduceHiddenChunk", [matHidden] * num_layers, lambda t: t.wgmma("reduce", chunk, 64, Major.MN))
            layerg.addTma("storeIntermChunk", [matInterm] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
            layerg.addTma("storeGateOutChunk", [matGateOut] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
        layerg.addTma("loadGateTileChunk", [matGateOut] * num_layers, lambda t: t.wgmma_load(chunk, 64, Major.MN))
        layerg.addTma("loadIntermTileChunk", [matInterm] * num_layers, lambda t: t.wgmma_load(chunk, 64, Major.MN))
        layerg.addTma("storeSiluChunk", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
    else:
        layerg.addTma("loadRMSLayer", [matRMSHidden] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
        layerg.addTma("reduceHiddenLayer", [matHidden] * num_layers, lambda t: t.wgmma("reduce", chunk, 64, Major.MN))
        layerg.addTma("loadSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
        layerg.addTma("loadAttnOLayer", [attnO] * num_layers, lambda t: t.wgmma_load(chunk, gemv_input_tile, Major.K))
        layerg.addTma("storeInterm", [matInterm] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
        layerg.addTma("storeGateOut", [matGateOut] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
        layerg.addTma("reduceInterm", [matInterm] * num_layers, lambda t: t.wgmma("reduce", chunk, 64, Major.MN))
        layerg.addTma("reduceGateOut", [matGateOut] * num_layers, lambda t: t.wgmma("reduce", chunk, 64, Major.MN))
        layerg.addTma("storeSiluLayer", [matSiLUOut] * num_layers, lambda t: t.wgmma_store(chunk, 64, Major.MN))
    layerg.addTma("loadRMSInputW", matRMSInputW[1:], lambda t: t.tensor1d("load", hidden))
    layerg.addTma("loadRMSPostAttnW", matRMSPostAttnW, lambda t: t.tensor1d("load", hidden))
    layerg.addTma("loadOutWs", matOutWs, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadDown", matDowns, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadUp", matUps, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadGate", matGates, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadQW", matqWs, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadKW", matkWs, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))
    layerg.addTma("loadVW", matvWs, lambda t: t.wgmma_load(gemv_tile_m, gemv_tile_k, Major.K))

    matQ_attn_views = [attnQ[:chunk].view(chunk, num_kv_heads, head_group_size, head_dim) for attnQ in attnQs]
    matK_attn_views = [attnK.view(1, args.prefill_len, num_kv_heads, head_dim) for attnK in attnKs]
    matV_attn_views = [attnV.view(1, args.prefill_len, num_kv_heads, head_dim) for attnV in attnVs]
    matO_attn_view = attnO[:chunk].view(chunk, num_kv_heads, head_group_size, head_dim)

    tma_builder_mn = partial(build_tma_wgmma_mn, iK=-3)
    cord_func_mn = partial(cord_func_MN_major, iK=-3)
    tma_builder_k = partial(build_tma_wgmma_k, iN=-3)
    cord_func_k = partial(cord_func_K_major, iN=-3)

    layerg.addTma("loadQ", matQ_attn_views, lambda t: t._build("load", head_dim, 64, tma_gqa_load_q, cord_gqa_load_q))
    layerg.addTma("loadK", matK_attn_views, lambda t: t._build("load", head_dim, kv_block_size, tma_builder_k, cord_func_k))
    layerg.addTma("loadV", matV_attn_views, lambda t: t._build("load", head_dim, kv_block_size, tma_builder_mn, cord_func_mn))

    dae.build_groups()

    loadHidden1D = TmaLoad1D(matHidden, bytes=hidden * 2)
    storeRMSHidden1D = TmaStore1D(matRMSHidden, bytes=hidden * 2)

    pre_attn_rms_init = SchedRMSShared(
        num_token=chunk,
        epsilon=eps,
        tmas=(TmaLoad1D(matRMSInputW[0]), loadHidden1D, storeRMSHidden1D),
        hidden_size=hidden,
    ).bar("output", layerg["bar_pre_attn_rms"]).place(chunk)
    pre_attn_rms_next = SchedRMSShared(
        num_token=chunk,
        epsilon=eps,
        tmas=(layerg["loadRMSInputW"].cord(0), loadHidden1D, storeRMSHidden1D),
        hidden_size=hidden,
    ).bar("input", layerg["bar_layer"]).bar("output", layerg.next("bar_pre_attn_rms")).place(chunk)
    post_attn_rms = SchedRMSShared(
        num_token=token_slots,
        epsilon=eps,
        tmas=(layerg["loadRMSPostAttnW"].cord(0), loadHidden1D, storeRMSHidden1D),
        hidden_size=hidden,
    ).bar("input", layerg["bar_out_mlp"]).bar("output", layerg["bar_post_attn_rms"]).place(token_slots)

    if chunk == 8:
        element_size = matHidden.element_size()
        regStoreQ = RegStore(0, size=chunk * 64 * element_size)
        regLoadQ = RegLoad(0)
        regStoreK = RegStore(1, size=chunk * 64 * element_size)
        regLoadK = RegLoad(1)

        q_proj = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(qw, chunk, hidden),
            tmas=(layerg["loadQW"], layerg["loadRMSLayer"], regStoreQ),
        ).bar("load", layerg["bar_pre_attn_rms"]).place(128)
        q_rope = SchedRope(
            ROPE_INTERLEAVE_512,
            tmas=(
                ToRopeTableCordAdapter(defaultg["loadRope"], 0, tile_repeats=head_dim // 64),
                regLoadQ,
                ToSplitMCordAdapter(layerg["storeQBlock"], qw // 64, 64),
            ),
        ).bar("store", layerg["bar_q_proj"]).place(128)
        k_proj = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(kw, chunk, hidden),
            tmas=(layerg["loadKW"], layerg["loadRMSLayer"], regStoreK),
        ).bar("load", layerg["bar_pre_attn_rms"]).place(64, base_sm=64)
        k_rope = SchedRope(
            ROPE_INTERLEAVE_512,
            tmas=(
                ToRopeTableCordAdapter(defaultg["loadRope"], 0, tile_repeats=head_dim // 64),
                regLoadK,
                ToPrefillKVStoreCordAdapter(layerg["storeKCacheBlock"], prefix_len, 64, num_m_tiles=kw // 64),
            ),
        ).bar("store", layerg["bar_qkv_attn"]).place(64, base_sm=64)
        v_proj = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(vw, chunk, hidden),
            tmas=(layerg["loadVW"], layerg["loadRMSLayer"], ToPrefillKVStoreCordAdapter(layerg["storeVCacheBlock"], prefix_len, 64)),
        ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"]).place(64)

        seq_lens = [prefix_len + idx + 1 for idx in range(chunk)]
        gqa = SchedAttentionTokenList(
            seq_lens=seq_lens,
            kv_block_size=kv_block_size,
            num_kv_heads=num_kv_heads,
            matO=matO_attn_view,
            tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
        ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"]).place(chunk * num_kv_heads)

        out_proj = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(hidden, chunk, hidden),
            tmas=(layerg["loadOutWs"], layerg["loadAttnOLayer"], layerg["reduceHiddenLayer"]),
        ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"]).place(128)

        gate_proj_low = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(4096, chunk, hidden),
            tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
        ).bar("load", layerg["bar_post_attn_rms"]).place(64)
        gate_proj_high = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=((4096, 2048), chunk, hidden),
            tmas=(layerg["loadGate"], layerg["loadRMSLayer"], layerg["storeGateOut"]),
        ).bar("store", layerg["bar_silu_in"]).place(32)
        up_proj_low = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(4096, chunk, hidden),
            tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
        ).bar("load", layerg["bar_post_attn_rms"]).place(64, base_sm=64)
        up_proj_high = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=((4096, 2048), chunk, hidden),
            tmas=(layerg["loadUp"], layerg["loadRMSLayer"], layerg["storeInterm"]),
        ).bar("store", layerg["bar_silu_in"]).place(32, base_sm=64)

        regStoreGate = RegStore(2, matGateOut[:, :64])
        regStoreUp = RegStore(3, matInterm[:, :64])
        gate_proj_fused = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=((6144, 8192), chunk, hidden),
            tmas=(layerg["loadGate"], layerg["loadRMSLayer"], regStoreGate),
        ).bar("load", layerg["bar_post_attn_rms"]).place(128)
        up_proj_fused = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=((6144, 8192), chunk, hidden),
            tmas=(layerg["loadUp"], layerg["loadRMSLayer"], regStoreUp),
        ).bar("load", layerg["bar_post_attn_rms"]).place(128)
        silu1 = SchedSmemSiLUInterleaved(
            num_token=chunk,
            gate_glob=matGateOut[:, :6144],
            up_glob=matInterm[:, :6144],
            out_glob=matSiLUOut[:, :6144],
        ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"]).place(chunk)
        silu_fused = SchedRegSiLUFused(
            num_token=chunk,
            store_tma=layerg["storeSiluLayer"],
            reg_gate=2,
            reg_up=3,
            base_offset=6144,
            stride=64,
        ).bar("output", layerg["bar_silu_out2"]).place(128)
        down_proj_low = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(hidden, chunk, 6144),
            tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
        ).bar("load", layerg["bar_silu_out1"]).place(128)
        down_proj_high = SchedGemv(
            Atom=Gemv_M64N8,
            MNK=(hidden, chunk, (6144, 8192)),
            tmas=(layerg["loadDown"], layerg["loadSiluLayer"], layerg["reduceHiddenLayer"]),
        ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"]).place(128)

        prologue_insts = [] if args.debug_stop_after == "attention" else [pre_attn_rms_init]

        if args.debug_stop_after == "attention":
            stage_insts = [gqa]
            for name in ("bar_layer", "bar_pre_attn_rms", "bar_q_proj", "bar_qkv_attn", "bar_out_mlp", "bar_post_attn_rms", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "out_proj":
            stage_insts = [
                q_proj,
                q_rope,
                k_proj,
                k_rope,
                v_proj,
                gqa,
                out_proj,
            ]
            for name in ("bar_layer", "bar_out_mlp", "bar_post_attn_rms", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "post_attn_rms":
            stage_insts = [
                q_proj,
                q_rope,
                k_proj,
                k_rope,
                v_proj,
                gqa,
                out_proj,
                post_attn_rms,
            ]
            for name in ("bar_layer", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "silu":
            stage_insts = [
                q_proj,
                q_rope,
                k_proj,
                k_rope,
                v_proj,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
            ]
            layerg.bindBarrier("bar_layer", 0)
        elif args.debug_stop_after == "down_proj":
            stage_insts = [
                q_proj,
                q_rope,
                k_proj,
                k_rope,
                v_proj,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
                down_proj_low,
                down_proj_high,
            ]
            layerg.bindBarrier("bar_layer", 0)
        else:
            stage_insts = [
                q_proj,
                q_rope,
                k_proj,
                k_rope,
                v_proj,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
                down_proj_low,
                down_proj_high,
                pre_attn_rms_next,
            ]
        final_kvq_stage = []
    else:
        element_size = matHidden.element_size()

        def make_q_block(block_idx: int):
            token_base = block_idx * 8
            base_sm = 64 * (block_idx % 2)
            q_proj_block = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(qw, 8, hidden),
                tmas=(
                    layerg["loadQW"],
                    ToTokenBlockNCordAdapter(layerg["loadRMSBlock"], token_base),
                    RegStore(0, size=8 * 64 * element_size),
                ),
            )
            q_rope_block = SchedRope(
                ROPE_INTERLEAVE_512,
                tmas=(
                    ToRopeTableCordAdapter(defaultg["loadRope"], block_idx, tile_repeats=head_dim // 64),
                    RegLoad(0),
                    ToPrefillKVStoreCordAdapter(layerg["storeQBlock"], token_base, 64, num_m_tiles=qw // 64),
                ),
            )
            return ListSchedule([q_proj_block, q_rope_block], lead_bars={"load"}, tail_bars={"store"}).bar(
                "load", layerg["bar_pre_attn_rms"]
            ).bar("store", layerg["bar_q_proj"]).place(64, base_sm=base_sm)

        def make_k_block(block_idx: int):
            token_base = block_idx * 8
            base_sm = 16 * (block_idx % 8)
            k_proj_block = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(kw, 8, hidden),
                tmas=(
                    layerg["loadKW"],
                    ToTokenBlockNCordAdapter(layerg["loadRMSBlock"], token_base),
                    RegStore(1, size=8 * 64 * element_size),
                ),
            )
            k_rope_block = SchedRope(
                ROPE_INTERLEAVE_512,
                tmas=(
                    ToRopeTableCordAdapter(defaultg["loadRope"], block_idx, tile_repeats=head_dim // 64),
                    RegLoad(1),
                    ToPrefillKVStoreCordAdapter(
                        layerg["storeKCacheBlock"],
                        prefix_len + token_base,
                        64,
                        num_m_tiles=kw // 64,
                    ),
                ),
            )
            return ListSchedule([k_proj_block, k_rope_block], lead_bars={"load"}, tail_bars={"store"}).bar(
                "load", layerg["bar_pre_attn_rms"]
            ).bar("store", layerg["bar_qkv_attn"]).place(16, base_sm=base_sm)

        def make_v_block(block_idx: int):
            token_base = block_idx * 8
            base_sm = 16 * (block_idx % 8)
            return SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(vw, 8, hidden),
                tmas=(
                    layerg["loadVW"],
                    ToTokenBlockNCordAdapter(layerg["loadRMSBlock"], token_base),
                    ToPrefillKVStoreCordAdapter(layerg["storeVCacheBlock"], prefix_len + token_base, 64),
                ),
            ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"]).place(16, base_sm=base_sm)

        q_blocks = [make_q_block(i) for i in range(num_q_blocks)]
        k_blocks = [make_k_block(i) for i in range(num_q_blocks)]
        if chunk >= 64:
            v_front = SchedGemm(
                Atom=Gemm_M64N64K64,
                MNK=(chunk, vw, hidden),
                tmas=(
                    layerg["loadRMSWide"],
                    layerg["loadVWide"],
                    ToTokenBlockNCordAdapter(layerg["storeVCacheWide"], prefix_len),
                ),
            ).bar("load", layerg["bar_pre_attn_rms"]).bar("store", layerg["bar_qkv_attn"]).place(16)
        else:
            v_front = ListSchedule([make_v_block(i) for i in range(num_q_blocks)])

        seq_lens = [prefix_len + idx + 1 for idx in range(chunk)]
        gqa = SchedAttentionTokenList(
            seq_lens=seq_lens,
            kv_block_size=kv_block_size,
            num_kv_heads=num_kv_heads,
            matO=matO_attn_view,
            tmas=(layerg["loadQ"], layerg["loadK"], layerg["loadV"]),
        ).bar("q", layerg["bar_q_proj"]).bar("k", layerg["bar_qkv_attn"]).bar("o", layerg["bar_attn_out"]).place(128)

        if chunk < 64:
            out_proj = SchedGemm(
                Atom=Gemm_M64N64K64,
                MNK=(64, hidden, hidden),
                tmas=(layerg["loadAttnOWide"], layerg["loadOutWide"], layerg["reduceHiddenWide"]),
            ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"]).place(64)
            gate_proj_low = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(4096, chunk, hidden),
                tmas=(layerg["loadGate"], layerg["loadRMSChunk"], layerg["storeGateOutChunk"]),
            ).bar("load", layerg["bar_post_attn_rms"]).place(64)
            gate_proj_high = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=((4096, 2048), chunk, hidden),
                tmas=(layerg["loadGate"], layerg["loadRMSChunk"], layerg["storeGateOutChunk"]),
            ).bar("store", layerg["bar_silu_in"]).place(32)
            up_proj_low = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(4096, chunk, hidden),
                tmas=(layerg["loadUp"], layerg["loadRMSChunk"], layerg["storeIntermChunk"]),
            ).bar("load", layerg["bar_post_attn_rms"]).place(64, base_sm=64)
            up_proj_high = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=((4096, 2048), chunk, hidden),
                tmas=(layerg["loadUp"], layerg["loadRMSChunk"], layerg["storeIntermChunk"]),
            ).bar("store", layerg["bar_silu_in"]).place(32, base_sm=64)
            regStoreGate = RegStore(2, matGateOut[:chunk, :64])
            regStoreUp = RegStore(3, matInterm[:chunk, :64])
            gate_proj_fused = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=((6144, 8192), chunk, hidden),
                tmas=(layerg["loadGate"], layerg["loadRMSChunk"], regStoreGate),
            ).bar("load", layerg["bar_post_attn_rms"]).place(128)
            up_proj_fused = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=((6144, 8192), chunk, hidden),
                tmas=(layerg["loadUp"], layerg["loadRMSChunk"], regStoreUp),
            ).bar("load", layerg["bar_post_attn_rms"]).place(128)
            silu1 = SchedSmemSiLUInterleaved(
                num_token=chunk,
                gate_glob=matGateOut[:chunk, :6144],
                up_glob=matInterm[:chunk, :6144],
                out_glob=matSiLUOut[:chunk, :6144],
            ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"]).place(chunk)
            silu_fused = SchedRegSiLUFused(
                num_token=chunk,
                store_tma=layerg["storeSiluChunk"],
                reg_gate=2,
                reg_up=3,
                base_offset=6144,
                stride=64,
            ).bar("output", layerg["bar_silu_out2"]).place(128)
            down_proj_low = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(hidden, chunk, 6144),
                tmas=(layerg["loadDown"], layerg["loadSiluChunk"], layerg["reduceHiddenChunk"]),
            ).bar("load", layerg["bar_silu_out1"]).place(128)
            down_proj_high = SchedGemv(
                Atom=Gemv_M64N8,
                MNK=(hidden, chunk, (6144, 8192)),
                tmas=(layerg["loadDown"], layerg["loadSiluChunk"], layerg["reduceHiddenChunk"]),
            ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"]).place(128)
        else:
            if chunk == 64:
                out_proj = SchedGemm(
                    Atom=Gemm_M64N128K64,
                    MNK=(chunk, hidden, hidden),
                    tmas=(layerg["loadAttnOWide128"], layerg["loadOutWide128"], layerg["reduceHiddenWide128"]),
                ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"]).place(32)
            else:
                out_proj = SchedGemm(
                    Atom=Gemm_M64N64K64,
                    MNK=(chunk, hidden, hidden),
                    tmas=(layerg["loadAttnOWide"], layerg["loadOutWide"], layerg["reduceHiddenWide"]),
                    prefetch=False,
                    group=False,
                ).bar("load", layerg["bar_attn_out"]).bar("store", layerg["bar_out_mlp"]).place(32)
            gate_proj_low = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, 6144, hidden),
                tmas=(layerg["loadRMSWide128"], layerg["loadGateWide128"], layerg["storeGateWide128"]),
                prefetch=False,
                group=False,
            ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"]).place(48)
            gate_proj_high = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, (6144, 8192), hidden),
                tmas=(layerg["loadRMSWide128"], layerg["loadGateWide128"], layerg["storeGateWide128"]),
                prefetch=False,
                group=False,
            ).bar("store", layerg["bar_silu_in"]).place(64)
            up_proj_low = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, 6144, hidden),
                tmas=(layerg["loadRMSWide128"], layerg["loadUpWide128"], layerg["storeIntermWide128"]),
            ).bar("load", layerg["bar_post_attn_rms"]).bar("store", layerg["bar_silu_in"]).place(48)
            up_proj_high = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, (6144, 8192), hidden),
                tmas=(layerg["loadRMSWide128"], layerg["loadUpWide128"], layerg["storeIntermWide128"]),
            ).bar("store", layerg["bar_silu_in"]).place(64)
            gate_proj_fused = ListSchedule([])
            up_proj_fused = ListSchedule([])
            silu1 = SchedSmemSiLUInterleaved(
                num_token=chunk,
                gate_glob=matGateOut[:chunk, :6144],
                up_glob=matInterm[:chunk, :6144],
                out_glob=matSiLUOut[:chunk, :6144],
            ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out1"]).place(chunk)
            silu_fused = SchedSmemSiLUSw128(
                num_token=chunk,
                num_tiles=8192 // 64,
                tmas=(layerg["loadGateTileChunk"], layerg["loadIntermTileChunk"], layerg["storeSiluChunk"]),
                base_offset=6144,
            ).bar("input", layerg["bar_silu_in"]).bar("output", layerg["bar_silu_out2"]).place(128)
            down_proj_low = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, hidden, 6144),
                tmas=(layerg["loadSiluWide128"], layerg["loadDownWide128"], layerg["reduceHiddenWide128"]),
                prefetch=False,
                group=False,
            ).bar("load", layerg["bar_silu_out1"]).place(32)
            down_proj_high = SchedGemm(
                Atom=Gemm_M64N128K64,
                MNK=(chunk, hidden, (6144, 8192)),
                tmas=(layerg["loadSiluWide128"], layerg["loadDownWide128"], layerg["reduceHiddenWide128"]),
                prefetch=False,
                group=False,
            ).bar("load", layerg["bar_silu_out2"]).bar("store", layerg["bar_layer"]).place(32)

        prologue_insts = [] if args.debug_stop_after == "attention" else [pre_attn_rms_init]
        if args.debug_stop_after == "attention":
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
            ]
            for name in ("bar_layer", "bar_pre_attn_rms", "bar_out_mlp", "bar_post_attn_rms", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "out_proj":
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
                out_proj,
            ]
            for name in ("bar_layer", "bar_out_mlp", "bar_post_attn_rms", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "post_attn_rms":
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
                out_proj,
                post_attn_rms,
            ]
            for name in ("bar_layer", "bar_silu_in", "bar_silu_out1", "bar_silu_out2"):
                layerg.bindBarrier(name, 0)
        elif args.debug_stop_after == "silu":
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
            ]
            layerg.bindBarrier("bar_layer", 0)
        elif args.debug_stop_after == "down_proj":
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
                down_proj_low,
                down_proj_high,
            ]
            layerg.bindBarrier("bar_layer", 0)
        else:
            stage_insts = [
                *q_blocks,
                *k_blocks,
                v_front,
                gqa,
                out_proj,
                post_attn_rms,
                gate_proj_low,
                gate_proj_high,
                up_proj_low,
                up_proj_high,
                silu1,
                gate_proj_fused,
                up_proj_fused,
                silu_fused,
                down_proj_low,
                down_proj_high,
                pre_attn_rms_next,
            ]
        final_kvq_stage = []

    final_stage_insts = final_kvq_stage if stop_after_final_kvq else []
    dae.bind_late_barrier_counts(*prologue_insts, *stage_insts)

    if prologue_insts:
        dae.i(*prologue_insts)
    loop_tail = [] if args.debug_stop_after is not None else [
        LoopM.toNext(dae.copy_mptrs(), num_layers, resource_group=layerg),
        LoopC.toNext(dae.copy_cptrs(), num_layers),
    ]
    dae.i(*stage_insts, *loop_tail)
    dae.s()

    if args.instdump is not None:
        dump_insts(dae, args.instdump)
        return

    if args.write_compute_ops is not None:
        write_compute_operator_file(dae, args.write_compute_ops)
        if not (args.correctness or args.launch or args.bench):
            return

    init_hidden = model.model.embed_tokens(chunk_token_ids.unsqueeze(0))[0].detach().contiguous()

    def reset_chunk_state():
        matHidden.zero_()
        matHidden[:chunk].copy_(init_hidden)
        matRMSHidden.zero_()
        attnO.zero_()
        matInterm.zero_()
        matGateOut.zero_()
        matSiLUOut.zero_()
        for layer_idx in range(num_layers):
            attnQs[layer_idx].zero_()
        if attention_only_refs is not None:
            attnQs[0][:chunk].copy_(attention_only_refs["q_chunk"])
            attnKs[0][:prefix_len + chunk].copy_(attention_only_refs["k_full"])
            attnVs[0][:prefix_len + chunk].copy_(attention_only_refs["v_full"])

    if args.launch:
        reset_chunk_state()
        print(f"[launch] chunk_start={prefix_len}, chunk_size={chunk}, prefill_len={args.prefill_len}")
        dae.launch()
        torch.cuda.synchronize()

    if args.correctness:
        print(f"[correctness] prefix={prefix_len} chunk={chunk} total={args.prefill_len}")
        reset_chunk_state()
        dae.launch()
        torch.cuda.synchronize()

        with torch.no_grad():
            captured, output = reference_pass(model, build_inputs(seq_token_ids))

        all_ok = True
        layer_idx = 0
        q_ref = permute_rope_rows(
            captured[layer_idx]["q_proj"][0, prefix_len:prefix_len + chunk].contiguous(),
            head_dim,
            config.num_attention_heads,
        )
        k_ref = permute_rope_rows(
            captured[layer_idx]["k_proj"][0, prefix_len:prefix_len + chunk].contiguous(),
            head_dim,
            config.num_key_value_heads,
        )
        v_ref = captured[layer_idx]["v_proj"][0, prefix_len:prefix_len + chunk].contiguous()
        rope_chunk = rope_rows[:, None, :]
        q_ref_rope = apply_interleaved_rope(
            q_ref.view(chunk, config.num_attention_heads, head_dim),
            rope_chunk,
        ).reshape_as(q_ref)
        k_ref_rope = apply_interleaved_rope(
            k_ref.view(chunk, config.num_key_value_heads, head_dim),
            rope_chunk,
        ).reshape_as(k_ref)
        checks = [
            check_tensor_threshold(f"layer{layer_idx}_q_rope_chunk", q_ref_rope, attnQs[layer_idx][:chunk], 5.0),
            check_tensor_threshold(
                f"layer{layer_idx}_k_rope_cache_chunk",
                k_ref_rope,
                attnKs[layer_idx][prefix_len:prefix_len + chunk],
                5.0,
            ),
            check_tensor_threshold(f"layer{layer_idx}_v_cache_chunk", v_ref, attnVs[layer_idx][prefix_len:prefix_len + chunk], 5.0),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

        if num_layers == 1:
            manual_attn_core = manual_attention_chunk(
                q_chunk=attnQs[layer_idx][:chunk],
                k_full=attnKs[layer_idx][:prefix_len + chunk],
                v_full=attnVs[layer_idx][:prefix_len + chunk],
                head_group_size=head_group_size,
                prefix_len=prefix_len,
                chunk=chunk,
                head_dim=head_dim,
            )
            attn_core_checks = [
                check_tensor_threshold("layer0_attn_core_chunk", manual_attn_core, attnO[:chunk], 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in attn_core_checks)

        if args.debug_stop_after == "attention":
            ref_attn_core = manual_attention_chunk(
                q_chunk=q_ref,
                k_full=permute_rope_rows(
                    captured[layer_idx]["k_proj"][0, :prefix_len + chunk].contiguous(),
                    head_dim,
                    config.num_key_value_heads,
                ),
                v_full=captured[layer_idx]["v_proj"][0, :prefix_len + chunk].contiguous(),
                head_group_size=head_group_size,
                prefix_len=prefix_len,
                chunk=chunk,
                head_dim=head_dim,
            )
            ref_attn = F.linear(ref_attn_core, layers[layer_idx].self_attn.o_proj.weight.detach())
            dae_attn = F.linear(attnO[:chunk], layers[layer_idx].self_attn.o_proj.weight.detach())
            attn_checks = [
                check_tensor_threshold("layer0_attn_out_chunk", ref_attn, dae_attn, 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in attn_checks)
        elif args.debug_stop_after == "out_proj":
            out_proj_ref = captured[layer_idx]["post_attn_residual"][0, prefix_len:prefix_len + chunk].contiguous()
            out_proj_checks = [
                check_tensor_threshold("layer0_out_proj_residual_chunk", out_proj_ref, matHidden[:chunk], 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in out_proj_checks)
        elif args.debug_stop_after == "post_attn_rms":
            post_rms_ref = captured[layer_idx]["post_attn_rms"][0, prefix_len:prefix_len + chunk].contiguous()
            post_rms_checks = [
                check_tensor_threshold("layer0_post_attn_rms_chunk", post_rms_ref, matRMSHidden[:chunk], 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in post_rms_checks)
        elif args.debug_stop_after == "silu":
            gate_ref = captured[layer_idx]["gate_proj"][0, prefix_len:prefix_len + chunk].contiguous()
            up_ref = captured[layer_idx]["up_proj"][0, prefix_len:prefix_len + chunk].contiguous()
            silu_ref = (F.silu(captured[layer_idx]["gate_proj"][0, prefix_len:prefix_len + chunk].contiguous().float()) *
                        captured[layer_idx]["up_proj"][0, prefix_len:prefix_len + chunk].contiguous().float()).to(dtype)
            silu_checks = [
                check_tensor_threshold("layer0_gate_proj_low_chunk", gate_ref[:, :6144], matGateOut[:chunk, :6144], 8.0),
                check_tensor_threshold("layer0_up_proj_low_chunk", up_ref[:, :6144], matInterm[:chunk, :6144], 8.0),
                check_tensor_threshold("layer0_gate_proj_high_chunk", gate_ref[:, 6144:], matGateOut[:chunk, 6144:], 8.0),
                check_tensor_threshold("layer0_up_proj_high_chunk", up_ref[:, 6144:], matInterm[:chunk, 6144:], 8.0),
                check_tensor_threshold("layer0_silu_low_chunk", silu_ref[:, :6144], matSiLUOut[:chunk, :6144], 6.0),
                check_tensor_threshold("layer0_silu_high_chunk", silu_ref[:, 6144:], matSiLUOut[:chunk, 6144:], 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in silu_checks)
        elif args.debug_stop_after == "down_proj":
            down_proj_ref = captured[layer_idx]["hidden_state_out"][0, prefix_len:prefix_len + chunk].contiguous()
            down_proj_checks = [
                check_tensor_threshold("layer0_down_proj_chunk", down_proj_ref, matHidden[:chunk], 6.0),
            ]
            all_ok = all_ok and all(passed for passed, _ in down_proj_checks)
        else:
            final_layer_idx = num_layers - 1
            final_q_ref = permute_rope_rows(
                captured[final_layer_idx]["q_proj"][0, prefix_len:prefix_len + chunk].contiguous(),
                head_dim,
                config.num_attention_heads,
            )
            final_k_ref = permute_rope_rows(
                captured[final_layer_idx]["k_proj"][0, prefix_len:prefix_len + chunk].contiguous(),
                head_dim,
                config.num_key_value_heads,
            )
            final_v_ref = captured[final_layer_idx]["v_proj"][0, prefix_len:prefix_len + chunk].contiguous()
            final_q_ref_rope = apply_interleaved_rope(
                final_q_ref.view(chunk, config.num_attention_heads, head_dim),
                rope_chunk,
            ).reshape_as(final_q_ref)
            final_k_ref_rope = apply_interleaved_rope(
                final_k_ref.view(chunk, config.num_key_value_heads, head_dim),
                rope_chunk,
            ).reshape_as(final_k_ref)
            final_qkv_checks = [
                check_tensor_threshold("final_q_rope_chunk", final_q_ref_rope, attnQs[final_layer_idx][:chunk], 5.0),
                check_tensor_threshold(
                    "final_k_rope_cache_chunk",
                    final_k_ref_rope,
                    attnKs[final_layer_idx][prefix_len:prefix_len + chunk],
                    5.0,
                ),
                check_tensor_threshold(
                    "final_v_cache_chunk",
                    final_v_ref,
                    attnVs[final_layer_idx][prefix_len:prefix_len + chunk],
                    5.0,
                ),
            ]
            all_ok = all_ok and all(passed for passed, _ in final_qkv_checks)

        if not all_ok:
            raise RuntimeError("Prefill correctness check failed")
        print("[correctness] all checks passed")

    if args.bench:
        warmup = 1
        for _ in range(warmup):
            reset_chunk_state()
            dae.launch()
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(args.bench):
            reset_chunk_state()
            start = time.perf_counter()
            dae.launch()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1e3)

        mean_ms = sum(times_ms) / len(times_ms)
        per_token_ms = mean_ms / chunk
        toks_per_s = 1000.0 * chunk / mean_ms
        print(
            f"[bench] prefix={prefix_len} chunk={chunk} mean={mean_ms:.3f} ms "
            f"per_token={per_token_ms:.3f} ms toks_per_s={toks_per_s:.2f}"
        )
        print(f"[bench] samples_ms={', '.join(f'{t:.3f}' for t in times_ms)}")


if __name__ == "__main__":
    main()
