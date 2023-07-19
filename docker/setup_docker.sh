pip install ninja numba
git clone git@ipads.se.sjtu.edu.cn:gnn/collcachelib.git /ugache

wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz && \
  tar xf gurobi9.5.1_linux64.tar.gz && \
  mv gurobi951 /opt/gurobi-install

export GUROBI_HOME="/opt/gurobi-install/linux64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
export LIBRARY_PATH="${LIBRARY_PATH}:${GUROBI_HOME}/lib"
export C_INCLUDE_PATH="${GUROBI_HOME}/include"
export CPLUS_INCLUDE_PATH="${GUROBI_HOME}/include"

cd /ugache && \
  mkdir build && cd build && \
  cmake .. && make -j
