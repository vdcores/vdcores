# Makefile for DAE kernel (multi-file build)

# CUDA compiler
NVCC = nvcc
PYTHON ?= python

NVSHMEM_HOME ?= $(shell $(PYTHON) -c "import sys; from pathlib import Path; print(Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'nvidia' / 'nvshmem')")
NVSHMEM_INCLUDE_DIR := $(NVSHMEM_HOME)/include
NVSHMEM_LIBRARY_DIR := $(NVSHMEM_HOME)/lib
NVSHMEM_BUILD_DIR := build/nvshmem
NVSHMEM_RUNTIME_OBJECT := $(NVSHMEM_BUILD_DIR)/runtime.o
NVSHMEM_DLINK_OBJECT := $(NVSHMEM_BUILD_DIR)/runtime_dlink.o
# CUDA 13 diagnoses deprecated volatile syntax in NVSHMEM 3.4.5 collective
# headers even though this runtime does not instantiate those collectives.
NVSHMEM_HEADER_DIAGNOSTICS := -diag-suppress=3012,3013
# A nine-warp CTA places three warps on one of Hopper's four SM subpartitions,
# so each thread must fit the resulting 168-register ceiling.  Without this
# optional-build-only limit the NVSHMEM device link promotes the kernel to 254
# registers and the CTA cannot launch.  The EP WGMMA GEMV naturally fits this
# ceiling without spills; the standard eight-warp build remains unconstrained.
NVSHMEM_REGISTER_LIMIT := -maxrregcount=168

# CUDA architecture (adjust for your GPU)
# SM80 for A100, SM89 for H100, SM90 for Hopper
CUDA_ARCH = -gencode arch=compute_90a,code=sm_90a

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
	rm -rf $(APPS) $(TARGETS) build/generated $(NVSHMEM_BUILD_DIR)

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

$(NVSHMEM_RUNTIME_OBJECT): src/runtime.cu $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS) $(HEADERS)
	@test -f $(NVSHMEM_INCLUDE_DIR)/nvshmem.h
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -DDAE_ENABLE_NVSHMEM=1 \
		$(NVSHMEM_HEADER_DIAGNOSTICS) -I$(NVSHMEM_INCLUDE_DIR) \
		$(NVSHMEM_REGISTER_LIMIT) \
		-rdc=true -dc -Xcompiler -fPIC -o $@ $<

$(NVSHMEM_DLINK_OBJECT): $(NVSHMEM_RUNTIME_OBJECT)
	@test -f $(NVSHMEM_LIBRARY_DIR)/libnvshmem_device.a
	$(NVCC) $(CUDA_ARCH) -dlink -Xcompiler -fPIC \
		$(NVSHMEM_RUNTIME_OBJECT) \
		-L$(NVSHMEM_LIBRARY_DIR) -lnvshmem_device -o $@

%: $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS)

run: $(BIN)
	./$<

pyext: $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS) $(TARGETS)
	$(PYTHON) -m pip install -e . --no-build-isolation

# Build the device-linked DAE runtime and the small optional NVSHMEM allocation
# extension through the same setup.py. Host control remains in NVSHMEM4Py.
nvshmem-pyext: $(SELECTED_COMPUTE_OPS) $(COMPUTE_OPCODE_ORDER) $(DYNAMIC_COMPUTE_HANDLERS) $(NVSHMEM_RUNTIME_OBJECT) $(NVSHMEM_DLINK_OBJECT)
	DAE_ENABLE_NVSHMEM=1 \
	NVSHMEM_HOME=$(NVSHMEM_HOME) \
	DAE_RUNTIME_OBJECT=$(abspath $(NVSHMEM_RUNTIME_OBJECT)) \
	DAE_RUNTIME_DLINK_OBJECT=$(abspath $(NVSHMEM_DLINK_OBJECT)) \
	$(PYTHON) -m pip install -e . --no-build-isolation

FORCE:

.PHONY: all clean run FORCE nvshmem-pyext
