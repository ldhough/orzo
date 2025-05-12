CXX = clang++-17
# themachine
CXXFLAGS = -Wall -std=c++23 -mbmi -mbmi2 -mavx2 -static
# Zaratan
# CXXFLAGS = -Wall -std=c++23 -mbmi -mbmi2 -mavx512f -mavx512vl -mavx512bw -mavx2 -static

CXXFLAGS += -Iexternal/tlx # dep of pasta-toolbox/bit_vector
CXXFLAGS += -Iexternal/bit_vector/include # pasta-toolbox/bit_vector
CXXFLAGS += -Iexternal/utils/include
CXXFLAGS += -Iinclude

INCL = include/orzo

ifeq ($(DEBUG),1)
	CXXFLAGS += -DDEBUG -g -O0
else
	CXXFLAGS += -DNDEBUG -O3 -flto
endif

ifeq ($(CHECK_CORRECTNESS),1)
	CXXFLAGS += -DCHECK_CORRECTNESS
endif

all: orzo-benchmark

.PHONY: clean
clean:
	rm -f obj/*.o
	rm bin/orzo-benchmark

obj/comparison.o: benchmarking/comparison.cc $(INCL)/utils.h $(INCL)/bitvector.h $(INCL)/orzo.h
	$(CXX) $(CXXFLAGS) -c benchmarking/comparison.cc -o $@

orzo-benchmark: obj/comparison.o
	$(CXX) $(CXXFLAGS) -o bin/$@ $^
