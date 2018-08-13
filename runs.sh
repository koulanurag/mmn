#!/usr/bin/env bash

ENV=TomitaG-v0
GRU_SIZE=6
FILE=main_tomita_ternary.py
source ~/virtualenvs/rl/bin/activate

#python $FILE --gru_size $GRU_SIZE --bhx_size 4 --ox_size 0 --env $ENV --bhx_train &
#python $FILE --gru_size $GRU_SIZE --bhx_size 8 --ox_size 0 --env $ENV --bhx_train &
#python $FILE --gru_size $GRU_SIZE --bhx_size 16 --ox_size 0 --env $ENV --bhx_train &

#python $FILE --gru_size $GRU_SIZE --bhx_size 4 --ox_size 0 --env $ENV --bgru_train &
#python $FILE --gru_size $GRU_SIZE --bhx_size 8 --ox_size 0 --env $ENV --bgru_train &
#python $FILE --gru_size $GRU_SIZE --bhx_size 16 --ox_size 0 --env $ENV --bgru_train &
#

python $FILE --gru_size $GRU_SIZE --bhx_size 4 --ox_size 0 --env $ENV --generate_fsm &
python $FILE --gru_size $GRU_SIZE --bhx_size 8 --ox_size 0 --env $ENV --generate_fsm &
python $FILE --gru_size $GRU_SIZE --bhx_size 16 --ox_size 0 --env $ENV --generate_fsm &

exit

