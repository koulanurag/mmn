# Introduction
In this document, a manual on how to work with the **LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS**'s code is described.
Topics covered in this README file is as follows: [general documentation about the code](), [how to use the pretrained models](), [a summary of results](), [how to run the code step by step](#step-by-step-manual), and [how to run the code with the prepared script](#use-prepared-scripts).

### A summary of results
#### Mode Counter Environments(MCE)

|   Game	|   B<sub>h</sub>, B<sub>f</sub>    |   Fine-Tuning Score |  Before Minimization	|   After Minization    |
|:---------:|:---------------------------------:|:-------------------:|:-----------------------:|:---------------------:|
|   >       | >                                 |  Before(%), After(%)|&#124;H&#124;, &#124;O&#124;. Accuracy(%)| &#124;H&#124;, &#124;O&#124;. Accuracy(%)|
|   	|   4,4	|  0.98, 1 |    7, 5, 1 |  4, 4, 1 	|
|   Amnesia	|   4,8	|  0.99, 1 	|   7, 7, 1	|  4, 4, 1 	|
|   	|   8,4	|   1, -	|   6, 5, 1	|  4, 4, 1 	|
|   	|   8,8	|   0.99, 1	|   7, 7, 1	|  4, 4, 1 	|
|   	|   4,4	|  1, - 	|   12, 6, 1|  10, 1, 1	|
|   Blind	|   4,8	|  1, - 	|   12, 8, 1|  10, 1, 1	|
|   	|   8,4	|  1, - 	|   5, 6, 1|   10, 1, 1	|
|   	|   8,8	|  0.78, 1 	|   13, 8, 1|  10, 1, 1	|
|   	|   4,4	|  0.98, 0.98 	|   58, 5, 0.98|  50, 4, 0.98|
|   Tracker	|   4,8	|  0.99, 1 	|   23, 5, 1|   10, 4, 1|
|   	|   8,4	|  0.98, 1 	|   91, 5, 1|   10, 4, 1|
|   	|   8,8	|  0.99, 1 	|   85, 5, 1|   10, 4, 1|





### Step by step manual
In the first step, the given model should be tested to see if it gets relatively good results or not. This command is to be used to test how the model performs during the test:
<br/>`python main_atari.py --env **ENVIRONMENT** --gru_test --gru_size 32`

Before training the QBNs, the discrete data should be generated. This means that quantizing the continuous data into a discrete form. This can be done by the following command:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_bn_data --gru_size 32 --generate_max_steps 100`

Now, the hidden state QBN should be trained based on the previously trained RNN(**GRU**) model and using the data generated in the last step. To do this, the following command should be run to train the Bottleneck Hidden State(BHX) net:
<br/>`python main_atari.py --env **ENVIRONMENT** --bhx_train --bhx_size 64 --gru_size 32 --generate_max_steps 100`

After it's done, the model and plots will be saved here:
<br/>`results/Atari/**ENVIRONMENT**/gru_32_bhx_64/`

Now, it's time to test the new model that was trained using the QBNs(BHX net). Run the following command:
<br/>`python main_atari.py --env **ENVIRONMENT** --bhx_test --bhx_size 64 --gru_size 32 --generate_max_steps 100`

Now, the observation QBN should be trained. To do this, the following command should be run to train the Bottleneck Observation State(OX) net:
<br/>`python main_atari.py --env **ENVIRONMENT** --ox_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100`

After it's done, the model and plots will be saved here:
<br/>`results/Atari/**ENVIRONMENT**/gru_32_ox_100/`

When the OX net is trained, it can be tested. Run the following command to do so:
<br/>`python main_atari.py --env **ENVIRONMENT** --ox_test --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100`

Having the QBNs, it's time to fine-tune the RNN(**GRU**) model based on them. Following command would do that:
<br/>`python main_atari.py --env **ENVIRONMENT** --bgru_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100`

When the fine-tuning is done, model and plots will be saved here:
<br/>`results/Atari/**ENVIRONMENT**/gru_32_hx_(64,100)_bgru`

In this step the trained model in previous step is going to be tested by the following command:
<br/>`python main_atari.py --env **ENVIRONMENT** --bgru_test --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100`

Congrats, you've made it so far :). It is the final step. Here the final results will be converted into a finite state machine explanation text file. Run the following command for that:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_fsm --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100`

And the FSM explanation files will be saved as text files here:
<br/>`results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/`
<br/>In this directory two most important, files containing the observation space and hidden state space beبore minimization(in the file named as: "fsm.txt") and after minimization(in the file named as: "minimized_moore_machine.txt")


### Use prepared scripts

Instead of going through the step by step manual described above, one can use the prepared scripts. This script starts from testing a given GRU model on a defined environment and ends by generating the FSM. One can do this by simply running:
<br/>`sh run_atari.sh` 
