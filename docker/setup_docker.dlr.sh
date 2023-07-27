git clone --branch sosp23ae https://github.com/Jeffery-Song/HugeCTR.git /hugectr_dev

cd /hugectr_dev && git checkout a1b41d26 && git -c submodule."third_party/collcachelib".update=none submodule update --init --recursive --depth 1

cd /hugectr_dev/hierarchical_parameter_server && python setup.py build && python setup.py install
cd /hugectr_dev/sparse_operation_kit && python setup.py build && python setup.py install