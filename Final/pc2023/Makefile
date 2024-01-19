all : demo

.PHONY : demo

CC = icpc
CFLAGS = -qopenmp -O3 -g -w -fno-alias -qopt-report -qopt-report-phase=all -mbranches-within-32B-boundaries -falign-double \
         -inline-calloc -falign-loops=8 -xCORE-AVX2

demo:
	$(CC) -o demo main.cpp $(CFLAGS)

.PHONY : clean
	rm demo
# -O3 -fopenmp -march=native
