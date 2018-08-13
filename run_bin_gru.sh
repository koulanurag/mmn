#!/usr/bin/env bash

ENV=TomitaF-v0
FILE=main_tomita_ternary_gru.py
source ~/virtualenvs/rl/bin/activate

#
#python $FILE --gru_size 2 --bhx_size 0 --ox_size 0 --env TomitaC-v0 --bgru_train --gru_scratch --bx_scratch &
#python $FILE --gru_size 2 --bhx_size 0 --ox_size 0 --env TomitaD-v0 --bgru_train --gru_scratch --bx_scratch &
python $FILE --gru_size 2 --bhx_size 0 --ox_size 0 --env TomitaC-v0 --generate_fsm --gru_scratch --bx_scratch &
python $FILE --gru_size 2 --bhx_size 0 --ox_size 0 --env TomitaD-v0 --generate_fsm --gru_scratch --bx_scratch &

#python $FILE --gru_size 8 --bhx_size 0 --ox_size 0 --env $ENV --bgru_train --gru_scratch --bx_scratch &
#python $FILE --gru_size 16 --bhx_size 0 --ox_size 0 --env $ENV --bgru_train --gru_scratch --bx_scratch &
#python $FILE --gru_size 24 --bhx_size 0 --ox_size 0 --env $ENV --bgru_train --gru_scratch --bx_scratch &

#python $FILE --gru_size 8 --bhx_size 0 --ox_size 0 --env $ENV --bgru_test --gru_scratch --bx_scratch &
#python $FILE --gru_size 16 --bhx_size 0 --ox_size 0 --env $ENV --bgru_test --gru_scratch --bx_scratch &
#python $FILE --gru_size 24 --bhx_size 0 --ox_size 0 --env $ENV --bgru_test --gru_scratch --bx_scratch &

#python $FILE --gru_size 8 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &
#python $FILE --gru_size 16 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &
#python $FILE --gru_size 24 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &

#
#python $FILE --gru_size 1 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &
#python $FILE --gru_size 2 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &
#python $FILE --gru_size 4 --bhx_size 0 --ox_size 0 --env $ENV --generate_fsm --gru_scratch --bx_scratch &

exit

