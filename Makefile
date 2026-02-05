# PANTHEON Build System

# --- Compiler Configuration ---
HIPCC ?= hipcc
CXXFLAGS := -O3 -std=c++14 -DNDEBUG -Ikernels/common
BUILD_DIR := build

# Platform Detection
ifeq ($(PLATFORM), CUDA)
    COMPILER := nvcc
    # NVCC needs explicit architecture and treating .cpp as .cu
    CXXFLAGS += -x cu --gpu-architecture=sm_70 -Wno-deprecated-gpu-targets
else
    COMPILER := hipcc
    # HIPCC auto-detects architecture
    CXXFLAGS += -std=c++17 --offload-arch=native
endif

# --- Auto-Discovery Logic ---

# 1. Find all .cpp files inside kernels/ subdirectories
#    Result: kernels/hbm_write/hbm_write.cpp kernels/cache_latency/cache_latency.cpp ...
ALL_SRCS := $(shell find kernels -name "*.cpp")

# 2. Exclude anything in the 'common' directory or root headers
SRCS := $(filter-out kernels/common/%, $(ALL_SRCS))

# 3. Determine Target Binaries
#    We strip the directory path and extension to get the binary name.
#    Example: kernels/hbm_write/hbm_write.cpp -> build/hbm_write
BINS := $(patsubst kernels/%/%.cpp, $(BUILD_DIR)/%, $(SRCS))
# Handle cases where the cpp file might not match the folder name exactly, 
# or is just sitting in a folder. This is a fallback to flattened names.
# For PANTHEON, usually specific naming is used, but this is robust:
BINS := $(foreach src,$(SRCS),$(BUILD_DIR)/$(basename $(notdir $(src))))

# --- Targets ---

.PHONY: all clean directories

all: directories $(BINS)

# Create build directory
directories:
	@mkdir -p $(BUILD_DIR)

# Generic Rule: Build any binary from its corresponding source found in kernels tree
# The 'vpath' directive tells Make to look for .cpp files inside kernels/ and its subdirs
vpath %.cpp $(sort $(dir $(SRCS)))

$(BUILD_DIR)/%: %.cpp
	@echo "[BUILD] Compiling $< -> $@"
	@$(COMPILER) $(CXXFLAGS) $< -o $@

clean:
	@echo "[CLEAN] Removing build artifacts..."
	@rm -rf $(BUILD_DIR)

# Debug helper to see what Make found
debug:
	@echo "Found Sources: $(SRCS)"
	@echo "Target Bins:   $(BINS)"
