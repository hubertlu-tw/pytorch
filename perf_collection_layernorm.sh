#!/bin/bash
set -e
set -x
echo "======================== Original Implementation ========================"
python3 benchmark_full.py 2>&1 | tee og.txt
#runTracer.sh python3 benchmark_full.py 2>&1 | tee og.txt
echo "======================== ENABLE_APEX_GAMMABETA=1 ========================"
ENABLE_APEX_GAMMABETA=1 python3 benchmark_full.py 2>&1 | tee gammabeta.txt
#ENABLE_APEX_GAMMABETA=1 runTracer.sh python3 benchmark_full.py 2>&1 | tee gammabeta.txt
echo "======================== ENABLE_APEX_GAMMABETA=1 + ENABLE_APEX_GAMMABETA=1 ========================"
ENABLE_APEX_GRADINPUT=1 ENABLE_APEX_GAMMABETA=1 python3 benchmark_full.py 2>&1 | tee gammabeta_gradinput.txt
#ENABLE_APEX_GRADINPUT=1 ENABLE_APEX_GAMMABETA=1 runTracer.sh python3 benchmark_full.py 2>&1 | tee gammabeta_gradinput.txt
cho "======================== ENABLE_APEX_GAMMABETA=1 + ENABLE_APEX_GAMMABETA=1 + ENABLE_REFACTORED_BLOCKSIZE=1 ========================"
ENABLE_REFACTORED_BLOCKSIZE=1 ENABLE_APEX_GRADINPUT=1 ENABLE_APEX_GAMMABETA=1 python3 benchmark_full.py 2>&1 | tee gammabeta_gradinput_blocksize.txt
#ENABLE_REFACTORED_BLOCKSIZE=1 ENABLE_APEX_GRADINPUT=1 ENABLE_APEX_GAMMABETA=1 runTracer.sh python3 benchmark_full.py 2>&1 | tee gammabeta_gradinput_blocksize.txt
echo "======================== ENABLE_APEX_GAMMABETA=1 + ENABLE_APEX_GAMMABETA=1 + ENABLE_REFACTORED_BLOCKSIZE=1 + ENABLE_GRADINPUT_TUNING=1 ========================"
ENABLE_GRADINPUT_TUNING=1 ENABLE_APEX_GRADINPUT=1 ENABLE_APEX_GAMMABETA=1 python3 benchmark_full.py 2>&1 | tee gammabeta_gradinput_blocksize_gradinputtuning.txt
# - cuComputePartGradGammaBeta - ENABLE_APEX_GAMMABETA
# - cuComputeGradGammaBeta - ENABLE_APEX_GAMMABETA

# - cuComputeGradInput - ENABLE_APEX_GRADINPUT

# - refactored the block size in cuComputePartGradGammaBeta - ENABLE_REFACTORED_BLOCKSIZE
