# This is an example script for sequence of commands to run
# Usage : ./run_tomita.sh TomitaA-v0
#!/usr/bin/env bash

ENV=$1
GRU_SIZE=32
FILE=main_tomita.py
## No OX-Training required in this case as observation space is already discrete

python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --gru_train --generate_max_steps 1000
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --generate_bn_data --generate_max_steps 1000
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --bhx_train
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --bhx_test
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --bgru_train
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --bgru_test
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --generate_fsm
python $FILE --env $ENV --gru_size $GRU_SIZE --bhx_size 64 --evaluate_fsm

exit
