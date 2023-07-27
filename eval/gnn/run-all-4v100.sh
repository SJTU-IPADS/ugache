rm -rf **/run-logs
make -C figure11-4v100 run
make -C figure12-4v100 run
tar czf ../gnn-4v100.tar.gz .
md5sum ../gnn-4v100.tar.gz