#!/bin/bash

### Benchmarks for DCIC
### To run, get juicebox_tools.jar and the test data files:
###   wget http://hicfiles.s3.amazonaws.com/internal/juicebox_tools/8.5.16/juicebox_tools.jar
###   wget ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551552/suppl/GSM1551552_HIC003_merged_nodups.txt.gz
###   wget https://s3.amazonaws.com/pkerp/data/matrix_test/chrX_5KB_bins.tsv.gz

### First benchmark

echo "Create index for data at 1MB resolution, all chromosomes"
time java -Xmx8g -jar juicebox_tools.jar pre -r 1000000 -v -q 1 GSM1551552_HIC003_merged_nodups.txt.gz HIC003_1MB.hic hg19

echo "Create index for data at all resolutions, all chromosomes"
time java -Xmx8g -jar juicebox_tools.jar pre -v -q 1 GSM1551552_HIC003_merged_nodups.txt.gz HIC003.hic hg19

echo "Querying: 256x256 / 2048x2048 / slices / single resolution no norm"
java -Xmx8g -jar juicebox_tools.jar benchmark HIC003_1MB.hic NONE

echo "Querying: 256x256 / 2048x2048 / slices / single resolution balanced norm"
java -Xmx8g -jar juicebox_tools.jar benchmark HIC003_1MB.hic KR

echo "Querying: 256x256 / 2048x2048 / slices / multiple resolution no norm"
java -Xmx8g -jar juicebox_tools.jar benchmark HIC003.hic NONE

echo "Querying: 256x256 / 2048x2048 / slices / multiple resolution balanced norm"
java -Xmx8g -jar juicebox_tools.jar benchmark HIC003.hic KR

# repeated query without loading index of 2048x2048
echo "Query without loading index 2048x2048....ten times"
time ./straw KR HIC003.hic 1:20480000:40960000 1:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 2:20480000:40960000 2:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 3:20480000:40960000 3:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 4:20480000:40960000 4:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 5:20480000:40960000 5:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 6:20480000:40960000 6:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 7:20480000:40960000 7:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 8:20480000:40960000 8:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 9:20480000:40960000 9:20480000:40960000 BP 10000 > tmp.txt 
time ./straw KR HIC003.hic 10:20480000:40960000 10:20480000:40960000 BP 10000 > tmp.txt 

### Second benchmark

#transform to valid-pairs format with scores
#gunzip -c chrX_5KB_bins.tsv.gz | awk '{print 0,"X",$1*5000,0,0,"X",$2*5000,1,$3}' | gzip > chrX_5KB.txt.gz

echo "Create index for binned data"
time java -Xmx8g -jar juicebox_tools.jar pre -r 5000 -c X -v -d chrX_5KB.txt.gz chrX_5KB.hic hg19

echo "Querying: 256x256 / 2048x2048 / slices / no norm"
java -Xmx8g -jar juicebox_tools.jar -v benchmark chrX_5KB.hic NONE X

echo "Querying: 256x256 / 2048x2048 / slices / balanced norm"
java -Xmx8g -jar juicebox_tools.jar -v benchmark chrX_5KB.hic KR X

echo "Binary index to text"
time java -Xmx8g -jar juicebox_tools.jar dump observed NONE chrX_5KB.hic X X BP 5000 chrX_5KB_out.txt
