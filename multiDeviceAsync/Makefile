CC=clang++ -std=c++11 -O2 -fsanitize=thread

multiDevice: multiDevice.o io_service_pool.o
	$(CC) -o $@ $^ -pthread -lOpenCL -L${BOOST_ROOT}/lib -lboost_system -lboost_filesystem

multiDevice.o: multiDevice.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT}

io_service_pool.o: io_service_pool.cpp
	$(CC) -o $@ $< -c -I${BOOST_ROOT}

clean:
	rm -f multiDevice multiDevice.o io_service_pool.o
