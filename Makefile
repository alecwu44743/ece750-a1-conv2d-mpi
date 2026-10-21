# Build: make            (uses mpicxx from Open MPI)
# Tests: make test       (runs the unit tests with 1, 2, 3, 4 and 8 processes)
MPICXX   ?= mpicxx
MPIRUN   ?= mpirun
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
CXXFLAGS += -Iinclude

# Apple clang does not accept -march=native on every version; fall back if needed.
ifeq ($(shell $(MPICXX) -march=native -x c++ -E /dev/null >/dev/null 2>&1 || echo no),no)
CXXFLAGS := $(filter-out -march=native,$(CXXFLAGS)) -mcpu=native
endif

BIN  := bin
LIB  := build/conv2d_serial.o build/conv2d_mpi.o build/conv2d_dist.o build/io.o

all: $(BIN)/conv2d_run $(BIN)/conv2d_bench $(BIN)/conv2d_test

build/%.o: src/%.cpp include/*.hpp | build
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

build/test_conv2d.o: tests/test_conv2d.cpp include/*.hpp | build
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/conv2d_run: build/main.o $(LIB) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

$(BIN)/conv2d_bench: build/benchmark.o $(LIB) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

$(BIN)/conv2d_test: build/test_conv2d.o $(LIB) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $^ -o $@

build $(BIN):
	mkdir -p $@

test: $(BIN)/conv2d_test
	@for n in 1 2 3 4 8; do $(MPIRUN) --oversubscribe -np $$n $(BIN)/conv2d_test || exit 1; done

clean:
	rm -rf build $(BIN)

.PHONY: all test clean
