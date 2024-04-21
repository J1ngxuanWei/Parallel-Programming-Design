#test.sh
#!/bin/sh
#PBS -N simd

pssh -h $PBS_NODEFILE mkdir -p /home/s2113619 1>&2
scp master:/home/s2113619/data/562_1.txt /home/s2113619
scp master:/home/s2113619/data/562_2.txt /home/s2113619
scp master:/home/s2113619/data/1011_1.txt /home/s2113619
scp master:/home/s2113619/data/1011_2.txt /home/s2113619
scp master:/home/s2113619/data/2362_1.txt /home/s2113619
scp master:/home/s2113619/data/2362_2.txt /home/s2113619

scp master:/home/s2113619/simd /home/s2113619
pscp -h $PBS_NODEFILE master:/home/s2113619/simd /home/s2113619 1>&2
/home/s2113619/simd
