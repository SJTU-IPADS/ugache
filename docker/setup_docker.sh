pip install ninja numba
git clone git@ipads.se.sjtu.edu.cn:gnn/collcachelib.git /ugache

wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz && \
  tar xf gurobi9.5.1_linux64.tar.gz && \
  mv gurobi951 /opt/gurobi-install

cd /ugache && \
  mkdir build && cd build && \
  cmake .. && make -j
