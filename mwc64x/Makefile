CC=clang++ -std=c++11 -O2

mwc64x: mwc64x.o
	$(CC) -o $@ $^ -lOpenCL

mwc64x.o: mwc64x.cpp
	$(CC) -o $@ $< -c

clean:
	rm -f mwc64x mwc64x.o
