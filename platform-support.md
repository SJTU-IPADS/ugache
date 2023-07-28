# Supporting Other Platforms
UGache only natively supports the 3 mentioned platform in readme.
For other multi-GPU platforms with NVLink interconnects, we provides preliminary support in this document.
Please be aware that this preliminary support is not stable yet, requires significant manual configuration and testing, and could change at any time.
It has not been thoroughly tested, and may leads to sub-optimal performance.
We are still actively improving UGache' generalizability. Please feel free to contact us if you encounter any issue.

## Describing Platform
The multi-GPU topology is mainly described in adjacency array.
Prepare a description file in following format to describe your multi-GPU platform.
Below is an example of server A, the DGX V100 station:
```
4   # num gpu
350 # local HBM bandwidth in GB/s
8   # host PCIe bandwidth in GB/s
# NVLink adjacency array: line i lists GPU IDs that is connected to GPU_i
1 2 3 # e.g. GPU_0 is connected these 3 GPUs
2 3 0
3 0 1
0 1 2
# NVLink bandwidth array associated with the aforementioned adjacency array
38 38 38
38 38 38
38 38 38
38 38 38
```

## Measuring Bandwidth
All bandwidth can be obtained via the this sample: [p2pBandwidthLatencyTest](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/p2pBandwidthLatencyTest). Here's an example output:
```bash
$ ./p2pBandwidthLatencyTest
                ...
P2P Connectivity Matrix
     D\D     0     1     2     3
     0	     1     1     1     1
     1	     1     1     1     1
     2	     1     1     1     1
     3	     1     1     1     1
                ...
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0 755.01  15.46  15.71  15.75
     1  15.47 782.22  15.71  15.75
     2  15.77  15.83 783.60  15.83
     3  15.76  15.79  15.84 783.99
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3
     0 761.45  96.87  96.91  96.87
     1  96.88 783.40  96.92  96.87
     2  96.92  96.87 784.39  96.87
     3  96.92  96.87  96.91 783.40
                ...
```

The local bandwidth comes from the diagonal in `Bidirectional P2P=Enabled Bandwidth Matrix`, with it's value halved.
The NVLink bandwidth also comes from halved `Bidirectional P2P=Enabled Bandwidth Matrix`.
The host bandwidth comes from halving the `Bidirectional P2P=Disabled Bandwidth Matrix`.
Note that these bandwidth may needs to be tuned lower, since the random access in embedding extraction may not reach the ideal bandwidth of sequential copy used in the sample.

## Run Tests
To enable this preliminary support, write the file and export its path as the environment variable `COLL_TOPO_FILE` before running any experiments:
```
COLL_TOPO_FILE=<path_to_topo_file> make run
```

## Examples
The above example comes from the DGX-V100 station with 4xV100. Here we provide another example on DGX-1 system with 8xV100:
```bash
$ ./p2pBandwidthLatencyTest
                            ...
P2P Connectivity Matrix
     D\D     0     1     2     3     4     5     6     7
     0	     1     1     1     1     0     0     1     0
     1	     1     1     1     1     0     0     0     1
     2	     1     1     1     1     1     0     0     0
     3	     1     1     1     1     0     1     0     0
     4	     0     0     1     0     1     1     1     1
     5	     0     0     0     1     1     1     1     1
     6	     1     0     0     0     1     1     1     1
     7	     0     1     0     0     1     1     1     1
                            ...
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 770.08   8.56  11.74  11.74  11.65  11.50  11.52  10.72
     1   8.48 771.41  11.69  11.78  11.56  11.50  11.41  10.80
     2  11.66  11.67 771.60   8.50  11.56  11.52  11.52  10.76
     3  11.62  11.63   8.69 771.60  11.64  11.55  11.50  10.85
     4  11.52  11.56  11.66  11.58 771.80   8.59  11.57  10.81
     5  11.45  11.57  11.54  11.50   8.52 770.27  11.74  10.75
     6  11.49  11.50  11.53  11.56  11.63  11.75 771.60   8.14
     7  10.92  10.75  10.79  10.82  10.74  10.65   8.16 771.60
                            ...
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5      6      7
     0 769.51  48.50  48.49  96.91  11.59  11.53  96.91  10.84
     1  48.50 770.27  96.91  48.49  11.63  11.55  11.58  96.90
     2  48.48  96.91 771.22  96.92  48.49  11.55  11.58  10.88
     3  96.90  48.49  96.92 769.89  11.57  48.49  11.53  10.94
     4  11.57  11.52  48.49  11.65 771.22  96.89  48.48  96.91
     5  11.56  11.49  11.61  48.49  96.89 771.60  96.90  48.49
     6  96.90  11.53  11.61  11.52  48.48  96.84 770.46  48.49
     7  10.94  96.90  10.76  10.90  96.85  48.49  48.49 770.84
                            ...
```

```
8   # num gpu
350 # local HBM bandwidth in GB/s
4   # host PCIe bandwidth in GB/s
# NVLink adjacency array: line i lists GPU IDs that is connected to GPU_i
3 6 1 2
7 2 3 0
1 3 0 4
2 0 5 1
5 7 2 6
6 4 7 3
0 5 4 7
4 1 6 5
# NVLink bandwidth array associated with the aforementioned adjacency array
38 38 19 19
38 38 19 19
38 38 19 19
38 38 19 19
38 38 19 19
38 38 19 19
38 38 19 19
38 38 19 19
```