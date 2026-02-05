# PANTHEON Build System
HIPCC ?= hipcc
CXXFLAGS := -O3 -std=c++14 -DNDEBUG -Ikernels/common
BUILD_DIR := build

ifeq ($(PLATFORM), CUDA)
    COMPILER := nvcc
    CXXFLAGS += -x cu --gpu-architecture=sm_70 -Wno-deprecated-gpu-targets
else
    COMPILER := hipcc
    CXXFLAGS += -std=c++17 --offload-arch=native
endif

TARGETS := $(BUILD_DIR)/hbm_write $(BUILD_DIR)/hbm_read $(BUILD_DIR)/cache_latency $(BUILD_DIR)/compute_virus

all: $(TARGETS)

$(BUILD_DIR)/hbm_write: kernels/hbm_write/hbm_write.cpp
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/hbm_read: kernels/hbm_read/hbm_read.cpp
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/cache_latency: kernels/cache_latency/cache_latency.cpp
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/compute_virus: kernels/compute_virus/compute_virus.cpp
	@mkdir -p $(BUILD_DIR)
	$(COMPILER) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)
