CC=clang -O2

deviceQuery: deviceQuery.o
	$(CC) -o $@ $^ -lOpenCL

%.o: %.cpp
	$(CC) -o $@ $< -c

clean:
	rm -f deviceQuery *.o
