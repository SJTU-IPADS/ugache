git clone --branch ugache-patch https://github.com/Jeffery-Song/HugeCTR.git /hugectr_dev

cd /hugectr_dev && git submodule update --init --recursive --depth 1

cd /hugectr_dev/hierarchical_parameter_server && python setup.py build && python setup.py install
cd /hugectr_dev/sparse_operation_kit && python setup.py build && python setup.py install

cd /ugache && git submodule update --init --recursive 3rdparty/json

cd /ugache/python_tf2/collcache_tf2 && \
  python setup.py build && python setup.py install