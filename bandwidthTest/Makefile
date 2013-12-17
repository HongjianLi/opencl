CC=clang -O2

bandwidthTest: bandwidthTest.o
	$(CC) -o $@ $^ -L${ICD_ROOT}/bin -L${AMDAPPSDKROOT}/lib/x86_64 -L${INTELOCLSDKROOT}/lib/x64 -lOpenCL

%.o: %.cpp
	$(CC) -o $@ $< -c -I${ICD_ROOT}/inc -I${AMDAPPSDKROOT}/include -I${INTELOCLSDKROOT}/include

clean:
	rm -f bandwidthTest *.o
