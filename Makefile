# Makefile for DAE kernel (multi-file build)

# CUDA compiler
NVCC = nvcc
PYTHON ?= python

# CUDA architecture (adjust for your GPU)
# SM80 for A100, SM89 for H100, SM90 for Hopper
CUDA_ARCH = -gencode arch=compute_90a,code=sm_90a

GENERATED_INCLUDE_DIR := build/generated
SELECTED_COMPUTE_OPS := $(GENERATED_INCLUDE_DIR)/dae/selected_compute_ops.inc
COMPUTE_DISPATCH := include/dae/compute_dispatch.cuh
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
	rm -rf $(APPS) $(TARGETS) build/generated

# Build the executable, this is wildcard rule for multiple targets
%: app/%.cu $(TARGETS) $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -o $@ $< $(TARGETS) $(LDFLAGS)

$(SELECTED_COMPUTE_OPS): FORCE $(COMPUTE_OP_GENERATOR) $(COMPUTE_DISPATCH)
	@mkdir -p $(dir $@)
	@set -e; \
	if [ -n "$(strip $(DAE_COMPUTE_OPS))" ]; then export DAE_COMPUTE_OPS='$(DAE_COMPUTE_OPS)'; fi; \
	if [ -n "$(strip $(DAE_COMPUTE_OPS_FILE))" ]; then export DAE_COMPUTE_OPS_FILE='$(DAE_COMPUTE_OPS_FILE)'; fi; \
	$(PYTHON) $(COMPUTE_OP_GENERATOR) --dispatch $(COMPUTE_DISPATCH) --output $@

runtime.o: src/runtime.cu $(SELECTED_COMPUTE_OPS) $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $@ $<

%: $(SELECTED_COMPUTE_OPS)

run: $(BIN)
	./$<

pyext: $(SELECTED_COMPUTE_OPS) $(TARGETS)
	pip install -e . --no-build-isolation

FORCE:

.PHONY: all clean run FORCE
