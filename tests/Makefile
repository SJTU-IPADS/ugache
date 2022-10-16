%.out: %.cc
	g++ $< -o $@ -std=c++14 -g -O2 -fPIC -Wall -L../build -lcoll_cache -I../coll_cache_lib -I/opt/gurobi-install/gurobi-latest/linux64/include -I/usr/local/cuda/include -I../3rdparty/CLI11/include -L/usr/local/cuda/lib64 -L/opt/gurobi-install/gurobi951/linux64/lib -L/opt/intel/oneapi/tbb/latest/lib/intel64/gcc4.8 -lcuda -pthread -lrt -lcudart -lgurobi95 -lgurobi_c++ -ltbb -fopenmp
