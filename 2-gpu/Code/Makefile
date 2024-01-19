CC	= g++
CFLAGS	= -O3 -std=c++11 
CUDA = nvcc
SRC = main.cpp

all: logVS

logVS: libtestcu.a
	$(CC) $(SRC) -ltestcu -L./ -lcudart -L/usr/local/cuda/lib64 -o $@ $(CFLAGS) 

libtestcu.a: CUDA_acc.cu
	$(CUDA) -lib $^ -o $@ 

.PHONY: clean

clean: 
	rm -f *.a
	rm -f *.o
	rm -f logVS