CC	= g++ -fopenmp -mavx512f
CFLAGS	= -O3 -std=c++11 

all: logVS

logVS: main.o
	$(CC) -o $@ $^ $(CFLAGS)

main.o: main.cpp
	$(CC) -c $(CFLAGS) $<

.PHONY: clean

clean: 
	rm -f *.o
	rm -f logVS
