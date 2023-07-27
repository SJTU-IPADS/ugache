rm -rf **/run-logs
make -C figure11-8a100 run
make -C figure11-8a100-fix-cache-rate run
make -C figure12-8a100 run
make -C figure12-8a100-fix-cache-rate run
make -C figure13 run
make -C figure14 run
make -C figure15 run
tar czf ../gnn-8a100.tar.gz .
md5sum ../gnn-8a100.tar.gz