from functools import partial
import torch
import copy
from math import sqrt
from typing import Optional
from collections import defaultdict
import torch.nn.functional as F
from dae.launcher import *
from dae.util import *
from utils import *

torch.manual_seed(0)

gpu = torch.device("cuda")

torch.manual_seed(0)
Q_SEQ_LEN, KV_SEQ_LEN = 16, 16
ATTN_PAD_SEQ_LEN = 64
HEAD_DIM = 128
# TODO(zijian): change to grouped query
NUM_REQ, NUM_HEAD = 1, 32
HID_DIM = 4096
INTERM_DIM = 12288
N_LAYER = 1

num_sms = 132
dae = Launcher(num_sms, device=gpu)

class Module:
    # this intend to represent a unit where you can clearly define the workload partition
    # so it's less useful for composition purpose
    _mid_cnt = 0
    _bar_id = 0
    _bar_id_2_cnt = defaultdict(int)

    def __init__(self, num_sms: int):
        self.num_sms = num_sms
        self.id = Module._mid_cnt
        Module._mid_cnt += 1
    
    @staticmethod
    def get_non_tma_bar_id(cnt):
        bar_id = Module._bar_id + config.max_tmas
        Module._bar_id_2_cnt[bar_id] += cnt
        Module._bar_id = (Module._bar_id + 1) % (config.max_bars - config.max_tmas)
        return bar_id

    @staticmethod
    def add_tma_bar(cnt, bar: MemoryInstruction):
        # currently each TMA is unique id so this is trivial
        # TODO(zijian): check opcode or bar flag ?
        tma_bar_id = bar.arg
        Module._bar_id_2_cnt[tma_bar_id] += cnt

    @staticmethod
    def get_bar_cnt(bar_id):
        if bar_id not in Module._bar_id_2_cnt:
            raise ValueError(f"bar id is not registered: {bar_id}")
        return Module._bar_id_2_cnt[bar_id]

    @staticmethod
    def create_glob_bar(bar_id):
        return GlobalBarrier(Module.get_bar_cnt(bar_id), bar_id)

    def task(self, sm: int):
        raise NotImplementedError()

class RMSNorm(Module):
    # apply rms norm inplace
    def __init__(
        self, 
        num_sms,
        input: torch.Tensor,
        input_bar: Optional[GlobalBarrier],
        dim, epsilon, num_tokens
    ):
        assert num_tokens % num_sms == 0, "num_tokens must be divisible by num_sms"
        assert dim == input.shape[-1], "dim must match input's last dimension (row major)"

        super().__init__(num_sms)
        self.input = input
        self.input_bar = input_bar
        self.dim = dim
        self.epsilon = epsilon
        self.num_tokens = num_tokens
        self.raw_addr_bar = Module.get_non_tma_bar_id(self.num_sms)
        if dim == 4096:
            self.kernel = RMS_NORM_F16_K_4096
        else:
            raise NotImplementedError(f"dim {dim} not supported in RMSNorm")

    def task(self, sm: int, compute_only: bool = False):
        row_offset = sm * (self.num_tokens // self.num_sms)
        matIn = self.input[row_offset] # only start addr is needed
        if compute_only:
            insts = [self.kernel(self.num_tokens // self.num_sms, self.epsilon)]
        else:
            insts = [] if self.input_bar is None else [self.input_bar]
            insts += [
                self.kernel(self.num_tokens // self.num_sms, self.epsilon),
                RawAddress(matIn, self.raw_addr_bar).group(),
            ]
        return insts
    
class GEMV(Module):
    def __init__(
        self, 
        num_sms,
        tInput: torch.Tensor,
        tWeight: torch.Tensor,
        tOutput: torch.Tensor,
        input_bar: Optional[GlobalBarrier],
        M, N, K,
        tileM, tileK,
        batch,
        stage,
    ):
        if M // tileM >= num_sms and stage > 1:
            raise ValueError(f"M should be big enough to saturate all SMs: M={M}, tileM={tileM}, stage={stage}, num_sms={num_sms}")
        # assume all MN major
        assert N == 16, "assume GEMV for now"
        super().__init__(num_sms)
        self.input_bar = input_bar
        self.M = M
        self.N = N
        self.K = K
        self.tileM = tileM
        self.tileK = tileK
        self.batch = batch
        self.stage = stage

        self.loadA = TmaTensor(dae, tWeight).wgmma_load(self.tileM, self.tileK, Major.K)
        self.loadB = TmaTensor(dae, tInput).wgmma_load(self.N, self.tileK * self.batch, Major.K)
        if self.stage > 1:
            assert tileM == Gemv_M64N16.MNK[0]
            assert tileK == Gemv_M64N16.MNK[2]
            self.storeC = TmaTensor(dae, tOutput).wgmma("reduce", self.N, self.tileM, Major.MN)
        else:
            assert tileM == Gemv_M128N16.MNK[0]
            assert tileK == Gemv_M128N16.MNK[2]
            self.storeC = TmaTensor(dae, tOutput).wgmma_store(self.N, self.tileM, Major.MN)
        Module.add_tma_bar(self.M // self.tileM * self.stage, self.storeC)
    
    def task(self, sm: int, compute_only: bool = False):
        insts = [] if self.input_bar is None else [self.input_bar]
        if self.stage > 1:
            m_offset = sm // self.stage
            stage_idx = sm % self.stage
            k_total = self.K // self.stage
            k_offset = stage_idx * k_total
            m = m_offset * self.tileM
            if compute_only:
                insts = [Gemv_M64N16(k_total // self.tileK)]
            else:
                insts += [
                    Gemv_M64N16(k_total // self.tileK),
                    RepeatM.on(k_total // self.tileK // self.batch,
                        [self.loadB.cord(0, k_offset).group(), self.loadB.cord2tma(0, self.batch * self.tileK)],
                        *[
                            [self.loadA.cord(m, k_offset + self.tileK * i).group(), self.loadA.cord2tma(0, self.batch * self.tileK)]
                            for i in range(self.batch)
                        ]
                    ),
                    self.storeC.cord(0, m).bar().group(),
                ]
        else:
            m_total = self.M // self.num_sms
            assert m_total % self.tileM == 0
            m_start = sm * m_total
            if compute_only:
                for m in range(m_start, m_start + m_total, self.tileM):
                    insts += [Gemv_M128N16(self.K // self.tileK)]
            else:
                for m in range(m_start, m_start + m_total, self.tileM):
                    insts += [
                        Gemv_M128N16(self.K // self.tileK),
                        RepeatM.on(self.K // self.tileK // self.batch,
                        [self.loadB.cord(0, 0).group(), self.loadB.cord2tma(0, self.batch * self.tileK)],
                        *[
                            [self.loadA.cord(m, self.tileK * i).group(), self.loadA.cord2tma(0, self.batch * self.tileK)]
                            for i in range(self.batch)
                        ]
                    ),
                    self.storeC.cord(0, m).bar().group(),
                ]
        return insts

    
class Attention(Module):
    """
    TODO(zijian): currently only test prefill without kv cache
    TODO(zijian): swap AB in attn to better handle Q < 64 ?
    """


    def __init__(
        self, 
        num_sms,
        tQKV: torch.Tensor,
        tOutput: torch.Tensor,
        tKV_cache: Optional[torch.Tensor],
        input_bar: Optional[GlobalBarrier],
        kv_seq_len,
        qTile, kvTile,
    ):
        assert tKV_cache is None, "test prefill only for now"
        super().__init__(num_sms)
        assert tQKV.shape[-1] == HEAD_DIM * 3 * NUM_HEAD, "tQKV last dim mismatch"
        self.tKV_cache = tKV_cache
        self.input_bar = input_bar
        self.kv_seq_len = kv_seq_len
        self.qTile = qTile
        self.kvTile = kvTile

        tQKV_attn_view = tQKV.view(NUM_REQ, ATTN_PAD_SEQ_LEN, NUM_HEAD, HEAD_DIM * 3)
        tOut_attn_view = tOutput.view(NUM_REQ, ATTN_PAD_SEQ_LEN, NUM_HEAD, HEAD_DIM)

        # assume input is shape [NUM_REQ, Q_SEQ_LEN, NUM_HEAD, HEAD_DIM * (3 or 1)]
        tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
        cord_func_K = partial(cord_func_K_major, iN=-3)
        tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
        cord_func_MN = partial(cord_func_MN_major, iK=-3)
        
        self.loadQ = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, qTile, tma_builder_K, cord_func_K)
        self.loadK_from_qkv = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, kvTile, tma_builder_K, cord_func_K)
        self.loadV_from_qkv = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, kvTile, tma_builder_MN, cord_func_MN)
        self.storeO = TmaTensor(dae, tOut_attn_view)._build("store", HEAD_DIM, qTile, tma_builder_K, cord_func_K)

        # incase we want to mask out workload
        n_qtile = (ATTN_PAD_SEQ_LEN + self.qTile - 1) // self.qTile
        Module.add_tma_bar(num_sms * n_qtile, self.storeO)

    def task(self, sm: int, compute_only: bool = False):
        head = sm % NUM_HEAD
        req = sm // NUM_HEAD

        insts = [] if self.input_bar is None else [self.input_bar]
        for q in range(0, ATTN_PAD_SEQ_LEN, self.qTile):
            if compute_only:
                insts += [
                    ATTENTION_M64N64K16_F16_F32_64_64_hdim(ATTN_PAD_SEQ_LEN, active_kv_len=self.kv_seq_len),
                ]
                continue
            insts += [
                ATTENTION_M64N64K16_F16_F32_64_64_hdim(ATTN_PAD_SEQ_LEN, active_kv_len=self.kv_seq_len),
                self.loadQ.cord(req, q, head, 0).group(),
                RepeatM.on(ATTN_PAD_SEQ_LEN // self.kvTile,
                    [self.loadK_from_qkv.cord(req, 0, head, HEAD_DIM).group(), self.loadK_from_qkv.cord2tma(0, self.kvTile, 0, 0)],
                    [self.loadV_from_qkv.cord(req, 0, head, HEAD_DIM * 2).group(), self.loadV_from_qkv.cord2tma(0, self.kvTile, 0, 0)],
                ),
                self.storeO.cord(req, q, head, 0).bar().group(),
            ]
        return insts

class SiluAndMul(Module):
    def __init__(
        self,
        num_sms,
        tInput: torch.Tensor,
        tOutput: torch.Tensor,
        input_bar: Optional[GlobalBarrier],
        num_tokens: int, # active tokens
        interm_dim: int,
    ):
        super().__init__(num_sms)
        self.tInput = tInput
        self.tOutput = tOutput
        self.input_bar = input_bar
        self.num_tokens = num_tokens
        self.workload = num_tokens * interm_dim // num_sms
        self.raw_addr_bar = Module.get_non_tma_bar_id(self.num_sms * 2)
        self.interm_dim = interm_dim
        if interm_dim == 12288:
            self.kernel = SILU_MUL_F16_K_16_12288
        else:
            raise NotImplementedError(f"interm_dim {interm_dim} not supported in SiluAndMul")

    def task(self, sm: int, compute_only: bool = False):
        offset = sm * self.workload % self.interm_dim
        row_start = self.workload * sm // self.interm_dim
        if compute_only:
            insts = [self.kernel(self.workload, offset)]
        else:
            insts = [] if self.input_bar is None else [self.input_bar]
            insts += [
                self.kernel(self.workload, offset),
                #TODO(zijian): load raw addr does not need to update bar
                RawAddress(self.tInput[row_start], self.raw_addr_bar).group(),
                RawAddress(self.tOutput[row_start], self.raw_addr_bar).group(),
            ]
        return insts

# fixed buffer
# TODO(zijian): NEED zerofy OP for GEMV
t_qkv_proj_out = torch.zeros(NUM_REQ * ATTN_PAD_SEQ_LEN, 3 * NUM_HEAD * HEAD_DIM, dtype=torch.float16, device=gpu)
t_attn_out = torch.zeros(NUM_REQ * ATTN_PAD_SEQ_LEN, NUM_HEAD * HEAD_DIM, dtype=torch.float16, device=gpu)
t_gate_up_proj_out = torch.zeros(NUM_REQ * Q_SEQ_LEN, 2 * INTERM_DIM, dtype=torch.float16, device=gpu)
t_silu_and_mul_out = torch.zeros(NUM_REQ * Q_SEQ_LEN, INTERM_DIM, dtype=torch.float16, device=gpu)

if False:

# qkv proj
    qkv_proj_load_input = TmaTensor(dae, t_input_hidden_state).wgmma_load(NUM_REQ * Q_SEQ_LEN, 128 * 4, Major.K)
    qkv_proj_store_out = TmaTensor(dae, t_qkv_proj_out[:NUM_REQ * Q_SEQ_LEN]).wgmma_store(NUM_REQ * Q_SEQ_LEN, 128, Major.MN)

# attention
    tQKV_attn_view = t_qkv_proj_out.view(NUM_REQ, ATTN_PAD_SEQ_LEN, NUM_HEAD, HEAD_DIM * 3)
    tOut_attn_view = t_attn_out.view(NUM_REQ, ATTN_PAD_SEQ_LEN, NUM_HEAD, HEAD_DIM)
    attn_tma_builder_K = partial(build_tma_wgmma_k, iN = -3)
    attn_cord_func_K = partial(cord_func_K_major, iN=-3)
    attn_tma_builder_MN = partial(build_tma_wgmma_mn, iK = -3)
    attn_cord_func_MN = partial(cord_func_MN_major, iK=-3)
    attn_loadQ = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, 64, attn_tma_builder_K, attn_cord_func_K)
    attn_loadK = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, 64, attn_tma_builder_K, attn_cord_func_K)
    attn_loadV = TmaTensor(dae, tQKV_attn_view)._build("load", HEAD_DIM, 64, attn_tma_builder_MN, attn_cord_func_MN)
    attn_storeQ = TmaTensor(dae, tOut_attn_view)._build("store", HEAD_DIM, 64, attn_tma_builder_K, attn_cord_func_K)

# post attn proj
    attn_out_proj_load_input = TmaTensor(dae, t_attn_out[:NUM_REQ * Q_SEQ_LEN]).wgmma_load(NUM_REQ * Q_SEQ_LEN, 256 * 4, Major.K)
    attn_out_proj_store_out = TmaTensor(dae, t_attn_out_proj_out).wgmma("reduce", NUM_REQ * Q_SEQ_LEN, 64, Major.MN)


class Qwen3Layer:
    def __init__(self):
        self.ops: list[Module] = []
        self.gate_up_proj_w = uniform_rand_scaled(2 * INTERM_DIM, HID_DIM, dtype=torch.float16, device=gpu, scale=0.1)
        self.down_proj_w = uniform_rand_scaled(HID_DIM, INTERM_DIM, dtype=torch.float16, device=gpu, scale=0.1)
        self.qkv_proj_w = uniform_rand_scaled(3 * HEAD_DIM * NUM_HEAD, HID_DIM, dtype=torch.float16, device=gpu, scale=0.1)
        self.attn_out_proj_w = uniform_rand_scaled(HID_DIM, HEAD_DIM * NUM_HEAD, dtype=torch.float16, device=gpu, scale=0.1)
    
    def reference_forward(self, input_hidden_state: torch.Tensor):
        # input_hidden_state = input_hidden_state.view(NUM_REQ * Q_SEQ_LEN, HID_DIM)
        # var = input_hidden_state.float().pow(2).mean(dim=-1, keepdim=True)
        # rms_norm_out = input_hidden_state * torch.rsqrt(var + 1.0).to(dtype=input_hidden_state.dtype)
        # return rms_norm_out.view(NUM_REQ, Q_SEQ_LEN, HID_DIM)

        qkv_proj_out = torch.matmul(input_hidden_state, self.qkv_proj_w.T)  # [NUM_REQ * Q_SEQ_LEN, HEAD_DIM * 3 * NUM_HEAD]
        # return qkv_proj_out
        qkv_proj_out = qkv_proj_out.view(NUM_REQ, Q_SEQ_LEN, NUM_HEAD, HEAD_DIM * 3)

        Q = qkv_proj_out[:, :, :, 0:HEAD_DIM].permute(0, 2, 1, 3)  # [NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM]
        K = qkv_proj_out[:, :, :, HEAD_DIM:HEAD_DIM*2].permute(0, 2, 1, 3)  # [NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM]
        V = qkv_proj_out[:, :, :, HEAD_DIM*2:HEAD_DIM*3].permute(0, 2, 1, 3)  # [NUM_REQ, NUM_HEAD, KV_SEQ_LEN, HEAD_DIM]
        S = torch.matmul(Q, K.transpose(-1, -2)) # [NUM_REQ, NUM_HEAD, Q_SEQ_LEN, KV_SEQ_LEN]
        P = torch.softmax(S, dim=-1)
        O = torch.matmul(P, V)  # [NUM_REQ, NUM_HEAD, Q_SEQ_LEN, HEAD_DIM]
        O = O.permute(0, 2, 1, 3).contiguous().view(NUM_REQ * Q_SEQ_LEN, HEAD_DIM * NUM_HEAD)
        # return O
        attn_out_proj_out = torch.matmul(O, self.attn_out_proj_w.T)  # [NUM_REQ * Q_SEQ_LEN, HID_DIM]
        return attn_out_proj_out

        # another rmsnorm here
        var2 = attn_out_proj_out.float().pow(2).mean(dim=-1, keepdim=True)
        attn_out_proj_out = attn_out_proj_out * torch.rsqrt(var2 + 1.0).to(dtype=attn_out_proj_out.dtype)
        # return attn_out_proj_out

        gate_up_proj_out = torch.matmul(attn_out_proj_out, self.gate_up_proj_w.T)  # [NUM_REQ * Q_SEQ_LEN, 2 * INTERM_DIM]
        # return gate_up_proj_out

        silu_and_mul_out = F.silu(gate_up_proj_out[..., :INTERM_DIM]) * gate_up_proj_out[..., INTERM_DIM:]  # [NUM_REQ * Q_SEQ_LEN, INTERM_DIM]
        # return silu_and_mul_out

        down_proj_out = torch.matmul(silu_and_mul_out, self.down_proj_w.T)  # [NUM_REQ * Q_SEQ_LEN, HID_DIM]
        return down_proj_out.view(NUM_REQ, Q_SEQ_LEN, HID_DIM)
        
    def build_task(self, layer_input_bar: Optional[GlobalBarrier], input_hidden_state: torch.Tensor, build_mem_only: bool):
        num_req, q_seq_len, _ = input_hidden_state.shape
        assert num_req == NUM_REQ and q_seq_len == Q_SEQ_LEN, "input shape mismatch"
        input_hidden_state = input_hidden_state.view(NUM_REQ * Q_SEQ_LEN, HID_DIM)

        # rms_norm = RMSNorm(
        #     num_sms=NUM_REQ * Q_SEQ_LEN,
        #     input=input_hidden_state,
        #     input_bar=layer_input_bar,
        #     dim=input_hidden_state.shape[-1],
        #     epsilon=1.0,
        #     num_tokens=NUM_REQ * Q_SEQ_LEN
        # )
        # self.ops.append(rms_norm)
        # if not build_mem_only:
        #     wait_rms_norm = Module.create_glob_bar(rms_norm.raw_addr_bar).group()
        # else:
        #     wait_rms_norm = None
        # return wait_rms_norm, input_hidden_state

        # Attention Start
        qkv_proj = GEMV(
            num_sms=96,
            tInput=input_hidden_state, 
            tWeight=self.qkv_proj_w,
            tOutput=t_qkv_proj_out[:NUM_REQ * Q_SEQ_LEN],
            # input_bar=wait_rms_norm,
            input_bar=None,
            M=t_qkv_proj_out.shape[-1],
            N=NUM_REQ * Q_SEQ_LEN,
            K=HID_DIM,
            tileM=128,
            tileK=128,
            batch=4,
            stage=1,
        )
        self.ops.append(qkv_proj)
        if not build_mem_only:
            wait_qkv_proj = Module.create_glob_bar(qkv_proj.storeC.arg).group()
        else:
            wait_qkv_proj = None
        # return wait_qkv_proj, t_qkv_proj_out

        attn = Attention(
            num_sms=NUM_HEAD * NUM_REQ,
            tQKV=t_qkv_proj_out,
            tOutput=t_attn_out,
            tKV_cache=None,
            input_bar=wait_qkv_proj,
            kv_seq_len=KV_SEQ_LEN,
            qTile=64,
            kvTile=64,
        )
        self.ops.append(attn)
        if not build_mem_only:
            wait_attn = Module.create_glob_bar(attn.storeO.arg).group()
        else:
            wait_attn = None
        # return wait_attn
        
        assert NUM_REQ == 1, "we assume first Q_SEQ_LEN tokens are valid, multiple REQ creates strided pattern"

        t_attn_out_proj_out = torch.zeros(NUM_REQ * Q_SEQ_LEN, HID_DIM, dtype=torch.float16, device=gpu)
        # TODO(zijian): support inplace wb ?
        attn_out_proj = GEMV(
            num_sms=128,
            tInput=t_attn_out[:NUM_REQ * Q_SEQ_LEN],
            tWeight=self.attn_out_proj_w,
            tOutput=t_attn_out_proj_out,
            input_bar=wait_attn,
            M=t_attn_out_proj_out.shape[-1],
            N=Q_SEQ_LEN * NUM_REQ,
            K=NUM_HEAD * HEAD_DIM,
            tileM=64,
            tileK=256,
            batch=4,
            stage=2,
        )
        self.ops.append(attn_out_proj)
        if not build_mem_only:
            wait_attn_out_proj = Module.create_glob_bar(attn_out_proj.storeC.arg).group()
        else:
            wait_attn_out_proj = None
        return wait_attn_out_proj, t_attn_out_proj_out

        post_attn_rms_norm = RMSNorm(
            num_sms=NUM_REQ * Q_SEQ_LEN,
            input=t_attn_out_proj_out,
            input_bar=wait_attn_out_proj,
            dim=t_attn_out_proj_out.shape[-1],
            epsilon=1.0,
            num_tokens=NUM_REQ * Q_SEQ_LEN
        )
        self.ops.append(post_attn_rms_norm)
        if not build_mem_only:
            wait_post_attn_rms_norm = Module.create_glob_bar(post_attn_rms_norm.raw_addr_bar).group()
        else:
            wait_post_attn_rms_norm = None
        # return wait_post_attn_rms_norm

        # MLP Start
        gate_up_proj = GEMV(
            num_sms=192, # this is logical SM number, physical SM will be mapped when schedule
            tInput=t_attn_out_proj_out,
            tWeight=self.gate_up_proj_w,
            tOutput=t_gate_up_proj_out,
            input_bar=wait_post_attn_rms_norm,
            M=t_gate_up_proj_out.shape[-1],
            N=NUM_REQ * Q_SEQ_LEN,
            K=HID_DIM,
            tileM=128,
            tileK=128,
            batch=4,
            stage=1,
        )
        self.ops.append(gate_up_proj)
        if not build_mem_only:
            wait_gate_up_proj = Module.create_glob_bar(gate_up_proj.storeC.arg).group()
        else:
            wait_gate_up_proj = None
        # return wait_gate_up_proj

        silu_and_mul = SiluAndMul(
            num_sms=48,
            tInput=t_gate_up_proj_out,
            tOutput=t_silu_and_mul_out,
            input_bar=wait_gate_up_proj,
            num_tokens=NUM_REQ * Q_SEQ_LEN,
            interm_dim=INTERM_DIM,
        )
        self.ops.append(silu_and_mul)
        if not build_mem_only:
            wait_silu_and_mul = Module.create_glob_bar(silu_and_mul.raw_addr_bar).group()
        else:
            wait_silu_and_mul = None
        # return wait_silu_and_mul

        t_down_proj_out = torch.zeros(NUM_REQ * Q_SEQ_LEN, HID_DIM, dtype=torch.float16, device=gpu)
        down_proj = GEMV(
            num_sms=128,
            tInput=t_silu_and_mul_out,
            tWeight=self.down_proj_w,
            tOutput=t_down_proj_out,
            input_bar=wait_silu_and_mul,
            M=t_down_proj_out.shape[-1],
            N=NUM_REQ * Q_SEQ_LEN,
            K=INTERM_DIM,
            tileM=64,
            tileK=256,
            batch=4,
            stage=2,
        )
        self.ops.append(down_proj)
        if not build_mem_only:
            wait_down_proj = Module.create_glob_bar(down_proj.storeC.arg).group()
        else:
            wait_down_proj = None
        
        # pad non-tma bar to be the same
        for _ in range(16 - 3):
            Module.get_non_tma_bar_id(0)
        return wait_down_proj, t_down_proj_out.view(NUM_REQ, Q_SEQ_LEN, HID_DIM)
    
class QwenModel:
    def __init__(self):
        self.layers = [Qwen3Layer() for _ in range(N_LAYER)]

    def get_schedule_plan(self, num_sms, input_hidden_state: torch.Tensor):
        layer_bar = None
        for i, layer in enumerate(self.layers):
            _b, input_hidden_state = layer.build_task(None, input_hidden_state, i!=0)
            if i == 0:
                layer_bar = _b

        def schedule(sm: int):
            insts = []
            for il, layer in enumerate(self.layers):
                for op in layer.ops:
                    for i in range(sm, op.num_sms, num_sms):
                        insts += op.task(i, il != 0)
            insts.append(layer_bar)
            insts += [LoopM(N_LAYER, 0, resource_group_shift=16)]
            return insts
        return schedule, input_hidden_state

    def reference_forward(self, input_hidden_state: torch.Tensor):
        input = input_hidden_state
        for layer in self.layers:
            input = layer.reference_forward(input)
        return input