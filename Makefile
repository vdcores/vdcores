# Makefile for DAE kernel (multi-file build)

# CUDA compiler
NVCC = nvcc
PYTHON ?= python

# Datacenter Blackwell uses architecture-accelerated tensor-core instructions,
# so both the virtual and real targets must carry the `a` suffix.  Keep this
# overrideable for Hopper regression builds and for SM103/B300 validation.
DAE_CUDA_ARCH ?= 100a
export DAE_CUDA_ARCH
CUDA_ARCH ?= -gencode arch=compute_$(DAE_CUDA_ARCH),code=sm_$(DAE_CUDA_ARCH)

GENERATED_INCLUDE_DIR := build/generated
SELECTED_COMPUTE_OPS := $(GENERATED_INCLUDE_DIR)/dae/selected_compute_ops.inc
COMPUTE_OPCODE_ORDER := $(GENERATED_INCLUDE_DIR)/dae/compute_opcode_order.inc
DYNAMIC_COMPUTE_HANDLERS := $(GENERATED_INCLUDE_DIR)/dae/dynamic_compute_handlers.inc
COMPUTE_OP_GENERATED_STAMP := $(GENERATED_INCLUDE_DIR)/dae/compute_ops.generated.stamp
COMPUTE_DISPATCH := include/dae/compute_dispatch.cuh
OPCODE_REGISTRY := include/dae/opcode.cuh.inc
COMPUTE_OP_GENERATOR := tools/generate_selected_compute_ops.py

# Compiler flags
# NVCC_FLAGS = -DNDEBUG -O3 -std=c++20 $(if $(profile),-DDAE_PROFILE) # --ptxas-options=--verbose

# Linker flags (add CUDA driver library for TMA support)
LDFLAGS = -lcuda -lcublas

NVCC_FLAGS = -O3 -Iinclude/dae -Iinclude -I$(GENERATED_INCLUDE_DIR) -std=c++20 -Xptxas=-v -use_fast_math
NVCC_FLAGS += -lineinfo

ifneq ($(m2c_legacy),)
	NVCC_FLAGS += -DDAE_M2C_OBSERVER_WAIT=0
endif

ifneq ($(track_profile),)
	NVCC_FLAGS += -DDAE_TRACK_PROFILE
endif

ifneq ($(aggregate_profile),)
	NVCC_FLAGS += -DDAE_AGGREGATE_PROFILE
	export DAE_AGGREGATE_PROFILE := 1
endif

ifneq ($(mxfp_timeline),)
	NVCC_FLAGS += -DDAE_TRACK_PROFILE -DDAE_TRACK_MXFP_TIMELINE
	export DAE_TRACK_PROFILE := 1
	export DAE_TRACK_MXFP_TIMELINE := 1
endif

ifneq ($(global_insts),)
	NVCC_FLAGS += -DDAE_LOAD_INSTRUCTIONS=0
	export DAE_GLOBAL_INSTRUCTIONS := 1
endif

ifneq ($(aux_slots),)
	NVCC_FLAGS += -DDAE_NUM_SLOTS=27 -DDAE_NUM_INSTS=192 -DDAE_DYNAMIC_SMEM_KB=219 -DDAE_PACKED_SWAP_ATTENTION_SCRATCH=1
	export DAE_AUX_SLOTS := 1
endif

ifneq ($(nvfp4_stages),)
	NVCC_FLAGS += -DDAE_NVFP4_UMMA_PIPELINE_STAGES=$(nvfp4_stages)
	export DAE_NVFP4_UMMA_PIPELINE_STAGES := $(nvfp4_stages)
endif

ifneq ($(nvfp4_scale_stages),)
	NVCC_FLAGS += -DDAE_NVFP4_SCALE_COPY_STAGES=$(nvfp4_scale_stages)
	export DAE_NVFP4_SCALE_COPY_STAGES := $(nvfp4_scale_stages)
endif

ifneq ($(num_insts),)
	NVCC_FLAGS += -DDAE_NUM_INSTS=$(num_insts)
	export DAE_NUM_INSTS := $(num_insts)
endif

ifneq ($(num_slots),)
	NVCC_FLAGS += -DDAE_NUM_SLOTS=$(num_slots)
	export DAE_NUM_SLOTS := $(num_slots)
endif

ifneq ($(dynamic_smem_kb),)
	NVCC_FLAGS += -DDAE_DYNAMIC_SMEM_KB=$(dynamic_smem_kb)
	export DAE_DYNAMIC_SMEM_KB := $(dynamic_smem_kb)
endif

ifneq ($(ffn_specialized),)
	NVCC_FLAGS += -DDAE_FFN_SPECIALIZED_KERNELS=1
	export DAE_FFN_SPECIALIZED_KERNELS := 1
endif

ifneq ($(dsv4_rope_metadata_offset_kb),)
	NVCC_FLAGS += -DDAE_DSV4_ROPE_METADATA_OFFSET_KB=$(dsv4_rope_metadata_offset_kb)
	export DAE_DSV4_ROPE_METADATA_OFFSET_KB := $(dsv4_rope_metadata_offset_kb)
endif

ifneq ($(mxfp_tma_scale_stages),)
	NVCC_FLAGS += -DDAE_MXFP4_MXFP8_TMA_SCALE_STAGES=$(mxfp_tma_scale_stages)
	export DAE_MXFP4_MXFP8_TMA_SCALE_STAGES := $(mxfp_tma_scale_stages)
endif

ifneq ($(mxfp_direct_tma),)
	NVCC_FLAGS += -DDAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA=1
	export DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA := 1
endif

ifneq ($(mxfp_gate_up_raw_umma),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_RAW_UMMA=$(mxfp_gate_up_raw_umma)
	export DAE_MXFP_GATE_UP_RAW_UMMA := $(mxfp_gate_up_raw_umma)
endif

ifneq ($(mxfp_gate_up_fixed_bulk_scale),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_FIXED_BULK_SCALE=$(mxfp_gate_up_fixed_bulk_scale)
	export DAE_MXFP_GATE_UP_FIXED_BULK_SCALE := $(mxfp_gate_up_fixed_bulk_scale)
endif

ifneq ($(mxfp_gate_up_subtile_scale_slots),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_SUBTILE_SCALE_SLOTS=$(mxfp_gate_up_subtile_scale_slots)
	export DAE_MXFP_GATE_UP_SUBTILE_SCALE_SLOTS := $(mxfp_gate_up_subtile_scale_slots)
endif

ifneq ($(mxfp_gate_up_direct_output),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_DIRECT_OUTPUT=$(mxfp_gate_up_direct_output)
	export DAE_MXFP_GATE_UP_DIRECT_OUTPUT := $(mxfp_gate_up_direct_output)
endif

ifneq ($(mxfp_gate_up_direct_activation),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_DIRECT_ACTIVATION=$(mxfp_gate_up_direct_activation)
	export DAE_MXFP_GATE_UP_DIRECT_ACTIVATION := $(mxfp_gate_up_direct_activation)
endif

ifneq ($(mxfp_gate_up_ldu_weight_ring),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_LDU_WEIGHT_RING=$(mxfp_gate_up_ldu_weight_ring)
	export DAE_MXFP_GATE_UP_LDU_WEIGHT_RING := $(mxfp_gate_up_ldu_weight_ring)
endif

ifneq ($(mxfp_gate_up_direct_activation_tiles),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_DIRECT_ACTIVATION_TILES=$(mxfp_gate_up_direct_activation_tiles)
	export DAE_MXFP_GATE_UP_DIRECT_ACTIVATION_TILES := $(mxfp_gate_up_direct_activation_tiles)
endif

ifneq ($(mxfp_gate_up_fixed_output_rows),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_FIXED_OUTPUT_ROWS=$(mxfp_gate_up_fixed_output_rows)
	export DAE_MXFP_GATE_UP_FIXED_OUTPUT_ROWS := $(mxfp_gate_up_fixed_output_rows)
endif

ifneq ($(mxfp_gate_up_fixed_bf16_epilogue),)
	NVCC_FLAGS += -DDAE_MXFP_GATE_UP_FIXED_BF16_EPILOGUE=$(mxfp_gate_up_fixed_bf16_epilogue)
	export DAE_MXFP_GATE_UP_FIXED_BF16_EPILOGUE := $(mxfp_gate_up_fixed_bf16_epilogue)
endif

# Directories
ifeq ($(debug),)
	NVCC_FLAGS += -DNDEBUG
else
	NVCC_FLAGS += -DDAE_DEBUG_PRINT=$(debug)
endif

TARGETS := runtime.o

# Target executable
CUFILES := $(wildcard app/*.cu)
APPS := $(patsubst app/%.cu,%,$(CUFILES))

# Source files
SOURCES = main.cu 

# Header files (for dependency tracking)
HEADERS = $(wildcard include/dae/*.cuh) $(wildcard include/task/*.cuh) $(wildcard include/dae/pipeline/*.cuh)

# for make <target> run
BIN ?= $(firstword $(filter-out run,$(MAKECMDGOALS)))

# Default target
all: pyext

# Clean build artifacts
clean:
	rm -rf $(APPS) $(TARGETS) build/generated

# Build the executable, this is wildcard rule for multiple targets
%: app/%.cu $(TARGETS) $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -o $@ $< $(TARGETS) $(LDFLAGS)

$(COMPUTE_OP_GENERATED_STAMP): FORCE $(COMPUTE_OP_GENERATOR) $(COMPUTE_DISPATCH) $(OPCODE_REGISTRY)
	@mkdir -p $(dir $@)
	@set -e; \
	if [ -n "$(strip $(DAE_COMPUTE_OPS))" ]; then export DAE_COMPUTE_OPS='$(DAE_COMPUTE_OPS)'; fi; \
	if [ -n "$(strip $(DAE_COMPUTE_OPS_FILE))" ]; then export DAE_COMPUTE_OPS_FILE='$(DAE_COMPUTE_OPS_FILE)'; fi; \
	$(PYTHON) $(COMPUTE_OP_GENERATOR) --dispatch $(COMPUTE_DISPATCH) --opcode-registry $(OPCODE_REGISTRY) --output $(SELECTED_COMPUTE_OPS) --opcode-output $(COMPUTE_OPCODE_ORDER) --dynamic-handlers-output $(DYNAMIC_COMPUTE_HANDLERS); \
	touch $@

$(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS): $(COMPUTE_OP_GENERATED_STAMP)

runtime.o: src/runtime.cu $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS) $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $@ $<

%: $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS)

run: $(BIN)
	./$<

pyext: $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS) $(TARGETS)
	$(PYTHON) -m pip install -e . --no-build-isolation

FORCE:

.PHONY: all clean run FORCE
