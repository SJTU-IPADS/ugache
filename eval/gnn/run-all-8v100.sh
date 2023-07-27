rm -rf **/run-logs
make -C figure11-8v100 run
make -C figure12-8v100 run
tar czf ../gnn-8v100.tar.gz .
md5sum gnn-8v100.tar.gz