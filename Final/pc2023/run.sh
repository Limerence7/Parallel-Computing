#!/bin/bash

#SBATCH --job-name=pm_test

#SBATCH --partition=compute

#SBATCH -n 32

#SBATCH -w computer02

module load hdacp-run

module load gcc/11.1.0

module load intel/compilervars

module load intel/2018

source /gpfs/software/intel2021/oneapi/setvars.sh

make clean

make

hdacp-run -t demo pc2023 solver.h Makefile

