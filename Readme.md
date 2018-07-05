# MMN

This is the implementation of Moore Machine Network(MMN) covered in paper "Toward Learning Finite State Representations of Recurrent Policy Networks" - [IJCAI/ECAI 2018 Workshop on
Explainable Artificial Intelligence (XAI) 2018](http://home.earthlink.net/~dwaha/research/meetings/faim18-xai/)

## Installation
 * Python 3.5+
 * [Pytorch](http://pytorch.org/) -'0.4.0'
 * [gym_x](https://github.com/koulanurag/gym_x)
 * To install dependencies:
    ```bash
    cd mmn
    pip install -r requirements.txt
    ```

## Usage :
* Argument description: ```python main_gold_rush.py --help```
* Examples :

    | Description  |Command     |
    |:-----------|:-----------|
    |Train GRU | ```python main_gold_rush.py --gru_train --no_cuda --env GoldRushRead-v0```|
    |Test GRU  | ```python main_gold_rush.py --gru_test --no_cuda --env GoldRushRead-v0``` |
    |Train BottleNeck network| ```python main_gold_rush.py --bx_train --env GoldRushRead-v0 --no_cuda``` |
    |Test BottleNeck network| ```python main_gold_rush.py --bx_test --env GoldRushRead-v0 --no_cuda``` |
    |Train Binary GRU network <br> (pre-trained gru and bottleneck network)|```python main_gold_rush.py --bgru_train --env GoldRushRead-v0 --no_cuda```|
    |Test Binary GRU network <br> (pre-trained gru and bottleneck network)| ```python main_gold_rush.py --bgru_test --env GoldRushRead-v0 --no_cuda```|
    |Generate Finite Moore Machine| ```python main_gold_rush.py --generate_fsm --no_cuda --env GoldRushRead-v0```|
    |Train Binary GRU network <br> (scratch gru and bottleneck network)|```python main_gold_rush.py --bgru_train --gru_scratch --bx_scratch --env GoldRushRead-v0 --no_cuda```|
    |Test Binary GRU network <br> (scratch gru and bottleneck network)|```python main_gold_rush.py --bgru_test --gru_scratch --bx_scratch --env GoldRushRead-v0 --no_cuda```|
    |Generate Finite Moore Machine <br> (scratch gru and bottleneck network)|```python main_gold_rush.py --generate_fsm --gru_scratch --bx_scratch --env GoldRushRead-v0 --no_cuda```|

