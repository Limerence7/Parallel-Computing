CC	= g++
CFLAGS	= -lm -O3 -std=c++11 
CUDA = nvcc
SRC = pivot.cpp

all: pivot

pivot: libtestcu.a
	$(CC) $(SRC) -ltestcu -L./ -lcudart -L/usr/local/cuda/lib64 -o $@ $(CFLAGS) 

libtestcu.a: CUDA_acc.cu
	$(CUDA) -O3 -lib $^ -o $@ 

.PHONY: clean

clean: 
	rm -f *.a
	rm -f *.o
	rm -f pivot