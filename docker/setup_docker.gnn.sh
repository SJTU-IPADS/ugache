git clone --depth 1 --branch 0.9.1 --recursive https://github.com/dmlc/dgl.git /dgl
git clone --branch sosp23ae https://github.com/Jeffery-Song/wholegraph.git /wholegraph
git clone --branch sosp23ae https://github.com/SJTU-IPADS/gnnlab.git /gnnlab
cd /gnnlab && git submodule update --init --recursive 3rdparty/parallel-hashmap 3rdparty/CLI11 

cd /dgl && \
  cmake -S . -B build -DUSE_CUDA=ON -DBUILD_TORCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_FP16=ON && \
  make -j -C build && \
  cd python && python setup.py build && python setup.py install

cd /wholegraph && \
  git checkout 916cf7a9 && \
  mkdir -p build && cd build && \
  cmake .. && \
  make -j

cd /gnnlab && \
  git checkout 5f0abe65 && \
  python setup.py build && python setup.py install

cd /ugache/python && \
  python setup.py build && python setup.py install
