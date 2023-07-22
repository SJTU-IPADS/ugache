git clone --branch sxn-dev git@ipads.se.sjtu.edu.cn:gnn/hugectr.git /hugectr_dev

cd /hugectr_dev && git -c submodule."third_party/collcachelib".update=none submodule update --init --recursive

cd /hugectr_dev/hierarchical_parameter_server && python setup.py build && python setup.py install
cd /hugectr_dev/sparse_operation_kit && python setup.py build && python setup.py install