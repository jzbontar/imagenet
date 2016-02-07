#!/bin/bash
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l walltime=120:00:00
#PBS -l mem=32GB
#PBS -j oe
#PBS -M jure.zbontar@gmail.com
 
module purge
module load cuda 
module load intel
module load opencv

export TERM=xterm

cp -r /scratch/jz1640/imagenet/db_hdd/ $PBS_JOBTMP
cd $HOME/devel/imagenet2
./main.lua -db_dir $PBS_JOBTMP/db_hdd $args
