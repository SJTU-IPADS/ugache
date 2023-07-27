rm -rf **/run-logs
make -C figure11-8a100 run
make -C figure12-8a100 run
make -C figure16 run
tar czf ../dlr-8a100.tar.gz .
md5sum ../dlr-8a100.tar.gz