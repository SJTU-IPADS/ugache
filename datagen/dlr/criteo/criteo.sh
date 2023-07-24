#!/bin/bash

g++ step_1_unique_keys.cc     -o step_1_unique_keys.out     -std=c++11 -g -O0 -fPIC -Wall -pthread -lrt -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0
g++ step_2_replace_keys.cc    -o step_2_replace_keys.out    -std=c++11 -g -O0 -fPIC -Wall -pthread -lrt -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0
g++ step_3_slice_processed.cc -o step_3_slice_processed.out -std=c++11 -g -O0 -fPIC -Wall -pthread -lrt -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0

# downloaded criteo raw data(day_0.gz ~ day_23.gz) should be placed in the raw dir
root_dir="/datasets_dlr/"
raw_dir="${root_dir}data-raw/criteo_tb/"
key_dir="${root_dir}data-raw/criteo_tb/"
processed_dir="${root_dir}processed/criteo_tb/"
echo "Processing data files under root directory ${root_dir}..."

start=0
end=23
concat_end=18

# 0. download dataset manually from:
#       https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
#       https://tianchi.aliyun.com/dataset/144736
for i in `seq ${start} ${end}`
do
  if [ ! -f "${raw_dir}day_${i}.gz" ]; then
    echo "Missing ${raw_dir}day_${i}.gz. Please download criteo dataset first!"
  fi
done

# 1. decompress the gz files
for i in `seq ${start} ${end}`
do
	echo "gzip -cd ${raw_dir}day_${i}.gz > ${raw_dir}day_${i}..."
	gzip -cd "${raw_dir}day_${i}.gz" > "${raw_dir}day_${i}"
done

# 2. generate unique keys
fnames=''
for i in `seq ${start} ${end}`
do
	fnames="${fnames} day_${i}"
done
echo "./step_1_unique_keys.out ${raw_dir} ${key_dir} ${fnames}..."
./step_1_unique_keys.out ${raw_dir} ${key_dir} ${fnames}  >> run.log

# 3. replace raw keys with continuous key
echo "./step_2_replace_keys.out ${raw_dir} ${processed_dir} ${key_dir} ${fnames}..."
./step_2_replace_keys.out ${raw_dir} ${processed_dir} ${key_dir} ${fnames}  >> run.log

# 4. split processed data file into .label, .dense and .sparse
touch "${processed_dir}day_concat.label"
touch "${processed_dir}day_concat.dense"
touch "${processed_dir}day_concat.sparse"
for i in `seq ${start} ${concat_end}`
do
	echo "./step_3_slice_processed.out ${processed_dir}day_${i} 13 26 1 ..."
	./step_3_slice_processed.out "${processed_dir}day_${i}" 13 26 1  >> run.log
    cat "${processed_dir}day_${i}.label" >> "${processed_dir}day_concat.label"; rm "${processed_dir}day_${i}.label"
    cat "${processed_dir}day_${i}.dense" >> "${processed_dir}day_concat.dense"; rm "${processed_dir}day_${i}.dense"
    cat "${processed_dir}day_${i}.sparse" >> "${processed_dir}day_concat.sparse"; rm "${processed_dir}day_${i}.sparse"
done

