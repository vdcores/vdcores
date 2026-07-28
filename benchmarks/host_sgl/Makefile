CXX ?= g++
NVCC ?= nvcc
CPPFLAGS ?=
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler=-fPIC,-Wall,-Wextra
LDFLAGS ?=
LDLIBS ?= -libverbs -lmlx5 -lcudart -lpthread

TARGET := libhost_sgl_verbs.so
OBJECTS := host_sgl_verbs.o host_sgl_ring.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -shared -o $@ $^ $(LDLIBS)

host_sgl_verbs.o: host_sgl_verbs.cc host_sgl_ring_abi.h ../../include/dae/pool_host_abi.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

host_sgl_ring.o: host_sgl_ring.cu host_sgl_ring_abi.h ../../include/dae/pool_host_abi.h
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJECTS)
