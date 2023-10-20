pip install ninja numba
git clone --branch sosp23ae https://github.com/SJTU-IPADS/ugache-artifacts.git /ugache

wget https://packages.gurobi.com/10.0/gurobi10.0.3_linux64.tar.gz && \
  tar xf gurobi10.0.3_linux64.tar.gz && \
  mv gurobi1003 /opt/gurobi-install && \
  rm gurobi10.0.3_linux64.tar.gz

cd /ugache && \
  mkdir build && cd build && \
  cmake .. && make -j
