NVCC ?= nvcc
MPICXX ?= mpicxx

NVSHMEM_HOME ?= $(CONDA_PREFIX)/lib/python3.13/site-packages/nvidia/nvshmem
CUDA_ARCH ?= -gencode=arch=compute_90a,code=sm_90a

MPI_CFLAGS := $(shell $(MPICXX) --showme:compile 2>/dev/null)

# Find CUDA runtime libs from the active conda/CUDA environment.
CUDA_LIBDIR := $(CONDA_PREFIX)/lib
CUDA_DEVRT_LIBDIR := $(CONDA_PREFIX)/lib

TARGET = main
SRC = main.cu

all: $(TARGET)

# 1. Compile CUDA source to relocatable device object.
main.o: $(SRC)
	$(NVCC) -O3 -std=c++17 -rdc=true -dc $(CUDA_ARCH) \
		-I$(NVSHMEM_HOME)/include \
		$(MPI_CFLAGS) \
		-c $(SRC) -o main.o

# 2. Device-link CUDA/NVSHMEM device code.
main_dlink.o: main.o
	$(NVCC) -dlink $(CUDA_ARCH) \
		main.o \
		-L$(NVSHMEM_HOME)/lib \
		-lnvshmem_device \
		-o main_dlink.o

# 3. Final host link with mpicxx
#    This lets the TACC MPI wrapper resolve MPI/HCOLL/UCX/RDMA dependencies.
$(TARGET): main.o main_dlink.o
	$(MPICXX) main.o main_dlink.o -o $(TARGET) \
		-L$(NVSHMEM_HOME)/lib \
		-lnvshmem_host -lnvshmem_device \
		-L$(CUDA_LIBDIR) \
		-lcudadevrt -lcudart \
		-ldl -lpthread \
		-Wl,-rpath,$(NVSHMEM_HOME)/lib \
		-Wl,-rpath,$(CUDA_LIBDIR) \
		-Wl,-rpath,$(HCOLL_LIBDIR) \
		-Wl,-rpath-link,$(HCOLL_LIBDIR) \
		-Wl,-rpath-link,/opt/apps/nvidia24/openmpi/5.0.5/lib \
		-Wl,-rpath-link,/opt/apps/ucx/1.20.0/lib \
		-Wl,-rpath-link,/opt/apps/ucc/1.7.0/lib \
		-Wl,-rpath-link,/usr/lib64 \
		-Wl,--allow-shlib-undefined


clean:
	rm -f $(TARGET) main.o main_dlink.o
