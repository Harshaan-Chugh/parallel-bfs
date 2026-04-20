# ============================================================
# Parallel BFS - Makefile
# Builds both linear-algebraic and graph-first BFS implementations
# ============================================================

NVCC      = nvcc
NVCCFLAGS = -O2 -arch=sm_80
# sm_80 = A100 (Perlmutter). Change if needed.

BUILD_DIR = build

# Targets
LINALG     = $(BUILD_DIR)/bfs_linalg
GRAPHFIRST = $(BUILD_DIR)/bfs_graphfirst

.PHONY: all clean linalg graphfirst

all: $(BUILD_DIR) linalg graphfirst

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

linalg: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(LINALG) linear-algebraic/bfs_linalg.cu

graphfirst: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(GRAPHFIRST) graph-first/bfs_graphfirst.cu

clean:
	rm -rf $(BUILD_DIR)
