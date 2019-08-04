# This is an example script for sequence of commands to run
# Usage : ./run_control.sh CartPole-v1 32 64 64
#!/usr/bin/env bash

ENV=$1
GRU_SIZE=$2
BHX_SIZE=$3
OX_SIZE=$4
FILE=main_control.py

# Assuming Pre-trained models exist
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --gru_test
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --generate_bn_data --generate_max_steps 1000
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --bhx_train
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --bhx_test
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --ox_train
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --ox_test
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --bgru_train
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --bgru_test
python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --generate_fsm
# python3 $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size $BHX_SIZE --ox_size $OX_SIZE --evaluate_fsm

exit
