# Makefile to compile each .cu into an executable with the same basename

CUFILES := $(wildcard *.cu)
EXES := $(CUFILES:.cu=)

NVCC ?= nvcc
# NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_90a -g -keep
NVCCFLAGS ?= -O3 -std=c++20 -arch=sm_90a

.PHONY: all clean

all: $(EXES)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -lcuda -o $@ $<

clean:
	rm -f $(EXES)

# for make <target> run
BIN ?= $(firstword $(filter-out run,$(MAKECMDGOALS)))

run: $(BIN)
	./$<