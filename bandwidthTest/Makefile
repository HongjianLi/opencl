CC=clang -O2

bandwidthTest: bandwidthTest.o
	$(CC) -o $@ $^ -lOpenCL

%.o: %.cpp
	$(CC) -o $@ $< -c

clean:
	rm -f bandwidthTest *.o
