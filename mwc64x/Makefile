CC=clang++ -std=c++11 -O2

mwc64x: mwc64x.o
	$(CC) -o $@ $^ -L${ICD_ROOT}/bin -L${AMDAPPSDKROOT}/lib/x86_64 -L${INTELOCLSDKROOT}/lib/x64 -lOpenCL

mwc64x.o: mwc64x.cpp
	$(CC) -o $@ $< -c -I${ICD_ROOT}/inc -I${AMDAPPSDKROOT}/include -I${INTELOCLSDKROOT}/include

clean:
	rm -f mwc64x mwc64x.o
