export GUROBI_HOME="/opt/gurobi-install/linux64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/ugache/build:${GUROBI_HOME}/lib"
export LIBRARY_PATH="${LIBRARY_PATH}:/ugache/build:${GUROBI_HOME}/lib"
export C_INCLUDE_PATH="/ugache:${GUROBI_HOME}/include"
export CPLUS_INCLUDE_PATH="/ugache:${GUROBI_HOME}/include"

git clone --depth 1 --branch 0.9.1 --recursive https://github.com/dmlc/dgl.git /dgl
git clone --branch sxn-dev git@ipads.se.sjtu.edu.cn:gnn/wholegraph.git /wholegraph
git clone --branch coll-cache-dev git@ipads.se.sjtu.edu.cn:gnn/samgraph.git /gnnlab
cd /gnnlab && git submodule update --init --recursive 3rdparty/parallel-hashmap 3rdparty/CLI11 

cd /dgl && \
  cmake -S . -B build -DUSE_CUDA=ON -DBUILD_TORCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_FP16=ON && \
  make -j -C build && \
  cd python && python setup.py build && python setup.py install

cd /wholegraph && \
  mkdir -p build && cd build && \
  cmake .. && \
  make -j
cd /gnnlab && \
  python setup.py build && python setup.py install

cd /ugache/python && \
  python setup.py build && python setup.py install
