pip install ninja numba
git clone --branch sosp23ae-sxn-dev git@ipads.se.sjtu.edu.cn:gnn/collcachelib.git /ugache

wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz && \
  tar xf gurobi9.5.1_linux64.tar.gz && \
  mv gurobi951 /opt/gurobi-install && \
  rm gurobi9.5.1_linux64.tar.gz

cd /ugache && \
  mkdir build && cd build && \
  cmake .. && make -j
