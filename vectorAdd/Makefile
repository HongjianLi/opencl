CC=clang -O2
ES=../embedSources.sh

vectorAdd: vectorAdd.o
	$(CC) -o $@ $^ -lOpenCL

vectorAdd.o: vectorAdd.cpp vectorAdd.h
	$(CC) -o $@ $< -c

vectorAdd.h: vectorAdd.cl
	$(ES) $^ > $@

clean:
	rm -f vectorAdd vectorAdd.o vectorAdd.h
