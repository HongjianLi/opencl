CC=clang++ -std=c++11 -O2 -fsanitize=thread

multiDevice: multiDevice.o io_service_pool.o
	$(CC) -o $@ $^ -pthread -L${BOOST_ROOT}/lib -lboost_system -lboost_filesystem -L${ICD_ROOT}/bin -L${AMDAPPSDKROOT}/lib/x86_64 -L${INTELOCLSDKROOT}/lib/x64 -lOpenCL

multiDevice.o: multiDevice.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT} -I${ICD_ROOT}/inc -I${AMDAPPSDKROOT}/include -I${INTELOCLSDKROOT}/include

io_service_pool.o: io_service_pool.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT}

clean:
	rm -f multiDevice multiDevice.o io_service_pool.o
