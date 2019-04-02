#!/usr/bin/env bash

ENV=$1
GRU_SIZE=32
FILE=main_atari.py

python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bhx_train &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bhx_test &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --ox_train &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --ox_test &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bgru_train &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --bgru_test &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --generate_fsm &
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 100 --evaluate_fsm &

exit

