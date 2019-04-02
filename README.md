# MMN
In this document, a manual on how to work with the **LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS**'s code is described.
Topics covered in this document:
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)

A paper for this work is available [here](https://openreview.net/pdf?id=S1gOpsCctm). Below is the BibTeX entry in case you want to cite it:
```
@inproceedings{koul2018learning,
    title={Learning Finite State Representations of Recurrent Policy Networks},
    author={Anurag Koul and Alan Fern and Sam Greydanus},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=S1gOpsCctm},
}
```
## Installation
* Python 3.5+
* Pytorch
* gym_x
* To install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
In this section, a guide on how to use the code is presented.

## Parameters
To run the code, there are several parameters that should be set. Below is a list of them:
```
usage: main_atari.py [-h] [--generate_train_data] [--generate_bn_data]
                     [--generate_max_steps GENERATE_MAX_STEPS] [--gru_train]
                     [--gru_test] [--gru_size GRU_SIZE] [--gru_lr GRU_LR]
                     [--bhx_train] [--ox_train] [--bhx_test] [--ox_test]
                     [--bgru_train] [--bgru_test] [--bhx_size BHX_SIZE]
                     [--bhx_suffix BHX_SUFFIX] [--ox_size OX_SIZE]
                     [--train_epochs TRAIN_EPOCHS] [--batch_size BATCH_SIZE]
                     [--bgru_lr BGRU_LR] [--gru_scratch] [--bx_scratch]
                     [--generate_fsm] [--evaluate_fsm]
                     [--bn_episodes BN_EPISODES] [--bn_epochs BN_EPOCHS]
                     [--no_cuda] [--env ENV] [--env_seed ENV_SEED]
                     [--result_dir RESULT_DIR]

GRU to FSM

optional arguments:
  -h, --help            show this help message and exit
  --generate_train_data
                        Generate Train Data
  --generate_bn_data    Generate Bottle-Neck Data
  --generate_max_steps GENERATE_MAX_STEPS
                        Maximum number of steps to be used for data generation
  --gru_train           Train GRU Network
  --gru_test            Test GRU Network
  --gru_size GRU_SIZE   No. of GRU Cells
  --gru_lr GRU_LR       No. of GRU Cells
  --bhx_train           Train bx network
  --ox_train            Train ox network
  --bhx_test            Test bx network
  --ox_test             Test ox network
  --bgru_train          Train binary gru network
  --bgru_test           Test binary gru network
  --bhx_size BHX_SIZE   binary encoding size
  --bhx_suffix BHX_SUFFIX
                        suffix fo bhx folder
  --ox_size OX_SIZE     binary encoding size
  --train_epochs TRAIN_EPOCHS
                        No. of training episodes
  --batch_size BATCH_SIZE
                        batch size used for training
  --bgru_lr BGRU_LR     Learning rate for binary GRU
  --gru_scratch         use scratch gru for BGRU
  --bx_scratch          use scratch bx network for BGRU
  --generate_fsm        extract fsm from fmm net
  --evaluate_fsm        evaluate fsm
  --bn_episodes BN_EPISODES
                        No. of episodes for generating data for Bottleneck
                        Network
  --bn_epochs BN_EPOCHS
                        No. of Training epochs
  --no_cuda             no cuda usage
  --env ENV             Name of the environment
  --env_seed ENV_SEED   Seed for the environment
  --result_dir RESULT_DIR
                        Directory Path to store results
```

By having the proper set of parameters, now you can run the ```main_atari.py``` code to start the process. For a step by step manual click [here](#step-by-step-manual).


### Use prepared scripts

Instead of going through the step by step manual described below, you can use the prepared scripts. This script starts from testing a given GRU model on a defined environment and ends by generating the FSM. Although the script is only for atari environments, but applying it to different setting is fairly similar and easy. You can do this by simply running:
```bash
sh run_atari.sh **ENVIRONMENT**
``` 


### Step by step manual
In the first step, the given model should be tested to see if it gets relatively good results or not. This command is to be used to test how the model performs during the test:
```bash
python main_atari.py --env **ENVIRONMENT** --gru_test --gru_size 32
```

Before training the QBNs, the discrete data should be generated. This means that quantizing the continuous data into a discrete form. This can be done by the following command:
```bash
python main_atari.py --env **ENVIRONMENT** --generate_bn_data --gru_size 32 --generate_max_steps 100
```

Now, the hidden state QBN should be trained based on the previously trained RNN(**GRU**) model and using the data generated in the last step. To do this, the following command should be run to train the Bottleneck Hidden State(BHX) net:
```bash
python main_atari.py --env **ENVIRONMENT** --bhx_train --bhx_size 64 --gru_size 32 --generate_max_steps 100
```

After it's done, the model and plots will be saved here:
```bash
results/Atari/**ENVIRONMENT**/gru_32_bhx_64/
```

Now, it's time to test the new model that was trained using the QBNs(BHX net). Run the following command:
```bash
python main_atari.py --env **ENVIRONMENT** --bhx_test --bhx_size 64 --gru_size 32 --generate_max_steps 100
```

Now, the observation QBN should be trained. To do this, the following command should be run to train the Bottleneck Observation State(OX) net:
```bash
python main_atari.py --env **ENVIRONMENT** --ox_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100
```

After it's done, the model and plots will be saved here:
```bash
results/Atari/**ENVIRONMENT**/gru_32_ox_100/
```

When the OX net is trained, it can be tested. Run the following command to do so:
```bash
python main_atari.py --env **ENVIRONMENT** --ox_test --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100
```

Having the QBNs, it's time to fine-tune the RNN(**GRU**) model based on them. Following command would do that:
```bash
python main_atari.py --env **ENVIRONMENT** --bgru_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100
```

When the fine-tuning is done, model and plots will be saved here:
```bash
results/Atari/**ENVIRONMENT**/gru_32_hx_(64,100)_bgru
```

In this step the trained model in previous step is going to be tested by the following command:
```bash
python main_atari.py --env **ENVIRONMENT** --bgru_test --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100
```

Congrats, you've made it so far :). It is the final step. Here the final results will be converted into a finite state machine explanation text file. Run the following command for that:
```bash
python main_atari.py --env **ENVIRONMENT** --generate_fsm --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100
```

And the FSM explanation files will be saved as text files here:

```bash
results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/
```
In this directory two most important, files containing the observation space and hidden state space beبore minimization(in the file named as: "fsm.txt") and after minimization(in the file named as: "minimized_moore_machine.txt")




### Using pretrained models
For results to be easily reproducible, previously trained GRU models on different environments have been provided. You can simply use them to train new QBNs and reproduce the results presented in the paper. Models are accessible through this directory: ```results/Atari/```. The GRU cell size can be determined from the models' path, e.i. if a model is saved in a folder named as ```gru_32```, then the GRU cell size is 32. 
Having the pretrained GRU model, you can go to [how to run the code step by step](#step-by-step-manual) to start training the QBNs.

## Results

### MCE
Presenting the Mode Counter Environments(MCE) results, number of states and observations of the MMs extracted from the MMNs both before and after minimization. Moore Machine extraction for MCE(table 1 in paper):

<table>
  <tr>
    <th align="center" rowspan="2">Game</th>
    <th align="center" rowspan="2">Bh</th>
    <th align="center" rowspan="2">Bf</th>
    <th align="center" colspan="2">Fine-Tuning Score</th>
    <th align="center" colspan="3">Before Minimization</th>
    <th align="center" colspan="3">After Minimization</th>
  </tr>
  <tr>
    <td align="center">Before(%)</td>
    <td align="center">After(%)</td>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Acc(%)</td>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Acc(%)</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Amnesia</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">98</td>
    <td align="center">100</td>
    <td align="center">7</td>
    <td align="center">5</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">8</td>
    <td align="center">99</td>
    <td align="center">100</td>
    <td align="center">7</td>
    <td align="center">7</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">4</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">6</td>
    <td align="center">5</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">8</td>
    <td align="center">99</td>
    <td align="center">100</td>
    <td align="center">7</td>
    <td align="center">7</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Blind</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">12</td>
    <td align="center">6</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">1</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">12</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">1</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">4</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">5</td>
    <td align="center">6</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">1</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">8</td>
    <td align="center">78</td>
    <td align="center">100</td>
    <td align="center">13</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">1</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td rowspan="4">Tracker</td>
    <td align="center">4</td>
    <td align="center">4</td>
    <td align="center">98</td>
    <td align="center">98</td>
    <td align="center">58</td>
    <td align="center">5</td>
    <td align="center">98</td>
    <td align="center">50</td>
    <td align="center">4</td>
    <td align="center">98</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">8</td>
    <td align="center">99</td>
    <td align="center">100</td>
    <td align="center">23</td>
    <td align="center">5</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">4</td>
    <td align="center">98</td>
    <td align="center">100</td>
    <td align="center">91</td>
    <td align="center">5</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">8</td>
    <td align="center">8</td>
    <td align="center">99</td>
    <td align="center">100</td>
    <td align="center">85</td>
    <td align="center">5</td>
    <td align="center">100</td>
    <td align="center">10</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
</table>

### Grammars
The below table presents the test results for the trained RNNs giving the accuracy over a test set of 100 strings drawn from the same distribution as used for training. Moore Machine extraction for Tomita grammar(table 2 in paper):

<table>
  <tr>
    <th align="center" rowspan="2">Grammar</th>
    <th align="center" rowspan="2">RNN Acc(%)</th>
    <th align="center" rowspan="2">Bh</th>
    <th align="center" colspan="2">Fine-Tuning Score</th>
    <th align="center" colspan="2">Before Minimization</th>
    <th align="center" colspan="2">After Minimization</th>
  </tr>
  <tr>
    <td align="center">Before(%)</td>
    <td align="center">After(%)</td>
    <td align="center">|H|</td>
    <td align="center">Acc(%)</td>
    <td align="center">|H|</td>
    <td align="center">Acc(%)</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">1</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">13</td>
    <td align="center">100</td>
    <td align="center">2</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">28</td>
    <td align="center">100</td>
    <td align="center">2</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">2</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">13</td>
    <td align="center">100</td>
    <td align="center">3</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">14</td>
    <td align="center">100</td>
    <td align="center">3</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">3</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">34</td>
    <td align="center">100</td>
    <td align="center">5</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">39</td>
    <td align="center">100</td>
    <td align="center">5</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">4</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">17</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">18</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">5</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">95</td>
    <td align="center">96</td>
    <td align="center">192</td>
    <td align="center">96</td>
    <td align="center">115</td>
    <td align="center">96</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">316</td>
    <td align="center">100</td>
    <td align="center">4</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">6</td>
    <td align="center">99</td>
    <td align="center">8</td>
    <td align="center">98</td>
    <td align="center">98</td>
    <td align="center">100</td>
    <td align="center">98</td>
    <td align="center">12</td>
    <td align="center">98</td>
  </tr>
  <tr>
    <td align="center">99</td>
    <td align="center">16</td>
    <td align="center">99</td>
    <td align="center">99</td>
    <td align="center">518</td>
    <td align="center">99</td>
    <td align="center">11</td>
    <td align="center">99</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">7</td>
    <td align="center">100</td>
    <td align="center">8</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">25</td>
    <td align="center">100</td>
    <td align="center">5</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">100</td>
    <td align="center">16</td>
    <td align="center">100</td>
    <td align="center">-</td>
    <td align="center">107</td>
    <td align="center">100</td>
    <td align="center">5</td>
    <td align="center">100</td>
  </tr>
</table>

### Control Tasks
More experiments on control tasks have been done. Results are presented in the following table:

<table>
  <tr>
    <th align="center" rowspan="2">Game(# of actions)</th>
    <th align="center" rowspan="2">Bh</th>
    <th align="center" rowspan="2">Bf</th>
    <th align="center" colspan="3">Before Minimization</th>
    <th align="center" colspan="3">After Minimization</th>
  </tr>
  <tr>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Score</td>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Score</td>
  </tr>
  <tr>
    <td align="center">Cart Pole(2)</td>
    <td align="center">8</td>
    <td align="center">8</td>
    <td align="center">10</td>
    <td align="center">27</td>
    <td align="center">500</td>
    <td align="center">5</td>
    <td align="center">25</td>
    <td align="center">500</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">Lunar Lander(4)</td>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">2550</td>
    <td align="center">2197</td>
    <td align="center">172</td>
    <td align="center">75</td>
    <td align="center">77</td>
    <td align="center">134</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">2194</td>
    <td align="center">1996</td>
    <td align="center">201</td>
    <td align="center">27</td>
    <td align="center">37</td>
    <td align="center">205</td>
  </tr>
</table>

### Atari
 This table shows the performance of the trained MMNs before and after finetuning for different combinations of B<sub>h</sub> and B<sub>f</sub>. A few more games investigated and the results are added to the table 3 of the paper:

<table>
  <tr>
    <th align="center" rowspan="2">Game(# of actions)</th>
    <th align="center" rowspan="2">RNN(score)</th>
    <th align="center" rowspan="2">Bh</th>
    <th align="center" rowspan="2">Bf</th>
    <th align="center" colspan="2">Fine-Tuning Score</th>
    <th align="center" colspan="3">Before Minimization</th>
    <th align="center" colspan="3">After Minimization</th>
  </tr>
  <tr>
    <td align="center">Before</td>
    <td align="center">After</td>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Score</td>
    <td align="center">|H|</td>
    <td align="center">|O|</td>
    <td align="center">Score</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Pong(3)</td>
    <td align="center" rowspan="4">21</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">20</td>
    <td align="center">21</td>
    <td align="center">380</td>
    <td align="center">374</td>
    <td align="center">21</td>
    <td align="center">4</td>
    <td align="center">12</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">20</td>
    <td align="center">21</td>
    <td align="center">373</td>
    <td align="center">372</td>
    <td align="center">21</td>
    <td align="center">3</td>
    <td align="center">10</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">20</td>
    <td align="center">21</td>
    <td align="center">383</td>
    <td align="center">373</td>
    <td align="center">21</td>
    <td align="center">3</td>
    <td align="center">12</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">20</td>
    <td align="center">21</td>
    <td align="center">379</td>
    <td align="center">371</td>
    <td align="center">21</td>
    <td align="center">3</td>
    <td align="center">11</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Freeway(3)</td>
    <td align="center" rowspan="4">21</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">21</td>
    <td align="center">-</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">21</td>
    <td align="center">-</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">21</td>
    <td align="center">-</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">21</td>
    <td align="center">-</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
    <td align="center">1</td>
    <td align="center">1</td>
    <td align="center">21</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Breakout(4)</td>
    <td align="center" rowspan="4">773</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">32</td>
    <td align="center">423</td>
    <td align="center">1898</td>
    <td align="center">1874</td>
    <td align="center">423</td>
    <td align="center">8</td>
    <td align="center">30</td>
    <td align="center">423</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">25</td>
    <td align="center">415</td>
    <td align="center">1888</td>
    <td align="center">1871</td>
    <td align="center">415</td>
    <td align="center">8</td>
    <td align="center">30</td>
    <td align="center">415</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">41</td>
    <td align="center">377</td>
    <td align="center">1583</td>
    <td align="center">1514</td>
    <td align="center">377</td>
    <td align="center">11</td>
    <td align="center">27</td>
    <td align="center">377</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">85</td>
    <td align="center">379</td>
    <td align="center">1729</td>
    <td align="center">1769</td>
    <td align="center">379</td>
    <td align="center">8</td>
    <td align="center">30</td>
    <td align="center">379</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Space Invaders(4)</td>
    <td align="center" rowspan="4">1820</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">520</td>
    <td align="center">1335</td>
    <td align="center">1495</td>
    <td align="center">1502</td>
    <td align="center">1335</td>
    <td align="center">8</td>
    <td align="center">29</td>
    <td align="center">1335</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">365</td>
    <td align="center">1235</td>
    <td align="center">1625</td>
    <td align="center">1620</td>
    <td align="center">1235</td>
    <td align="center">12</td>
    <td align="center">29</td>
    <td align="center">1235</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">390</td>
    <td align="center">1040</td>
    <td align="center">1563</td>
    <td align="center">1457</td>
    <td align="center">1040</td>
    <td align="center">12</td>
    <td align="center">35</td>
    <td align="center">1040</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">520</td>
    <td align="center">1430</td>
    <td align="center">1931</td>
    <td align="center">1921</td>
    <td align="center">1430</td>
    <td align="center">6</td>
    <td align="center">27</td>
    <td align="center">1430</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Bowling(6)</td>
    <td align="center" rowspan="4">60</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">60</td>
    <td align="center">-</td>
    <td align="center">49</td>
    <td align="center">1</td>
    <td align="center">60</td>
    <td align="center">33</td>
    <td align="center">1</td>
    <td align="center">60</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">60</td>
    <td align="center">-</td>
    <td align="center">49</td>
    <td align="center">1</td>
    <td align="center">60</td>
    <td align="center">33</td>
    <td align="center">1</td>
    <td align="center">60</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">60</td>
    <td align="center">-</td>
    <td align="center">26</td>
    <td align="center">1</td>
    <td align="center">60</td>
    <td align="center">24</td>
    <td align="center">1</td>
    <td align="center">60</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">60</td>
    <td align="center">-</td>
    <td align="center">26</td>
    <td align="center">1</td>
    <td align="center">60</td>
    <td align="center">24</td>
    <td align="center">1</td>
    <td align="center">60</td>
  </tr>
  <tr>
    <td align="center" rowspan="4">Boxing(18)</td>
    <td align="center" rowspan="4">100</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center">94</td>
    <td align="center">100</td>
    <td align="center">1173</td>
    <td align="center">1167</td>
    <td align="center">100</td>
    <td align="center">13</td>
    <td align="center">79</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">64</td>
    <td align="center">400</td>
    <td align="center">98</td>
    <td align="center">100</td>
    <td align="center">2621</td>
    <td align="center">2605</td>
    <td align="center">100</td>
    <td align="center">14</td>
    <td align="center">119</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">100</td>
    <td align="center">94</td>
    <td align="center">97</td>
    <td align="center">2499</td>
    <td align="center">2482</td>
    <td align="center">97</td>
    <td align="center">14</td>
    <td align="center">106</td>
    <td align="center">97</td>
  </tr>
  <tr>
    <td align="center">128</td>
    <td align="center">400</td>
    <td align="center">97</td>
    <td align="center">100</td>
    <td align="center">1173</td>
    <td align="center">1169</td>
    <td align="center">100</td>
    <td align="center">14</td>
    <td align="center">88</td>
    <td align="center">100</td>
  </tr>
  <tr>
    <td align="center">Chopper Command(18)</td>
    <td align="center">5300</td>
    <td align="center">64</td>
    <td align="center">100</td>
    <td align="center"></td>
    <td align="center">4000</td>
    <td align="center">3710</td>
    <td align="center">3731</td>
    <td align="center">4000</td>
    <td align="center">38</td>
    <td align="center">182</td>
    <td align="center">1890</td>
  </tr>
</table>