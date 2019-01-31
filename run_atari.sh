#!/usr/bin/env bash

ENV=$1
GRU_SIZE=32
FILE=main_atari.py
#source /home/danesh/Research/Learning_FSM_GRU/venv/bin/activate

#python $FILE --env $ENV --gru_size 32 --bhx_size 64 --ox_size 100 --gru_train
#wait

#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --ox_size 400 --bhx_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 400 --bhx_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size 400 --bhx_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size 400 --bhx_train &
#
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 100 --ox_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 200 --ox_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 400 --ox_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 600 --ox_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size 800 --ox_train &

python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  100  --bgru_train
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  100  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  400  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  400  --bgru_train &

#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  200  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size  64 --ox_size  400  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  400  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  400  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  400  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  600  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  600  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  600  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  600  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  800  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  800  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  800  --bgru_train &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  800  --bgru_train &
##
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  100  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  100  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  400  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  400  --generate_fsm &

#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  200  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  400  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  400  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  400  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  400  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  600  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  600  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  600  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  600  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64  --ox_size  800  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 128 --ox_size  800  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 256 --ox_size  800  --generate_fsm &
#python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 512 --ox_size  800  --generate_fsm &


exit

