%.out: %.cc
	g++ $< workspace_pool.cc -o $@ -std=c++14 -g -O2 -fPIC -Wall -Wl,-rpath=../build -L../build -lcoll_cache -I../coll_cache_lib -I/usr/local/cuda/include -I../3rdparty/CLI11/include -L/usr/local/cuda/lib64 -lcuda -pthread -lrt -lcudart -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0
