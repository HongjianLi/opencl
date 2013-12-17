CC=clang -O2
ES=../embedSources.sh

vectorAdd: vectorAdd.o
	$(CC) -o $@ $^ -L${ICD_ROOT}/bin -L${AMDAPPSDKROOT}/lib/x86_64 -L${INTELOCLSDKROOT}/lib/x64 -lOpenCL

vectorAdd.o: vectorAdd.cpp vectorAdd.h
	$(CC) -o $@ $< -c -I${ICD_ROOT}/inc -I${AMDAPPSDKROOT}/include -I${INTELOCLSDKROOT}/include

vectorAdd.h: vectorAdd.cl
	$(ES) $^ > $@

clean:
	rm -f vectorAdd vectorAdd.o vectorAdd.h
