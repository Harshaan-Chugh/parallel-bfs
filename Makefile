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

# MPI (CPU-only, no CUDA dependency)
MPICC    = cc
MPIFLAGS = -O2 -Wall -std=c99

MPI_1D = $(BUILD_DIR)/bfs_mpi_1d
MPI_2D = $(BUILD_DIR)/bfs_mpi_2d

.PHONY: all clean linalg graphfirst mpi mpi-1d mpi-2d

all: $(BUILD_DIR) linalg graphfirst

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

linalg: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(LINALG) linear-algebraic/bfs_linalg.cu

graphfirst: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $(GRAPHFIRST) graph-first/bfs_graphfirst.cu

mpi: $(BUILD_DIR) mpi-1d mpi-2d

mpi-1d: $(BUILD_DIR)
	$(MPICC) $(MPIFLAGS) -o $(MPI_1D) mpi/bfs_mpi_1d.c -lm

mpi-2d: $(BUILD_DIR)
	$(MPICC) $(MPIFLAGS) -o $(MPI_2D) mpi/bfs_mpi_2d.c -lm

clean:
	rm -rf $(BUILD_DIR)
