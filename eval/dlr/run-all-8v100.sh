rm -rf **/run-logs
make -C figure11-8v100 run
make -C figure12-8v100 run
tar czf ../dlr-8v100.tar.gz .
md5sum ../dlr-8v100.tar.gz