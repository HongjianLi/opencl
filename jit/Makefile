CC=clang -O2

jit: jit.o
	$(CC) -o $@ $^ -lOpenCL

%.o: %.cpp
	$(CC) -o $@ $< -c

clean:
	rm -f jit *.o
