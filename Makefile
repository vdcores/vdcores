# Makefile for DAE kernel (multi-file build)

# CUDA compiler
NVCC = nvcc

# CUDA architecture (adjust for your GPU)
# SM80 for A100, SM89 for H100, SM90 for Hopper
CUDA_ARCH = -gencode arch=compute_90a,code=sm_90a

# Compiler flags
# NVCC_FLAGS = -DNDEBUG -O3 -std=c++20 $(if $(profile),-DDAE_PROFILE) # --ptxas-options=--verbose

# Linker flags (add CUDA driver library for TMA support)
LDFLAGS = -lcuda -lcublas

NVCC_FLAGS = -O3 -Iinclude/dae -Iinclude -std=c++20 -Xptxas=-v -use_fast_math
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
	rm -rf $(APPS) $(TARGETS)

%.o: src/%.cu $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -Xcompiler -fPIC -c -o $@ $<

# Build the executable, this is wildcard rule for multiple targets
%: app/%.cu $(TARGETS) $(HEADERS)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) -o $@ $< $(TARGETS) $(LDFLAGS)

run: $(BIN)
	./$<

pyext: $(TARGETS)
	pip install -e . --no-build-isolation

.PHONY: all clean run
