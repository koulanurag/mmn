# Introduction
In this document, a manual on how to work with the **[LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS](https://openreview.net/pdf?id=S1gOpsCctm)**'s code is described.
Topics covered in this document:
* [how to use the pretrained models](#using-pretrained-models)
* [a summary of results](#a-summary-of-results)
* [general documentation about the code](#code-walk-through)
* [how to run the code step by step](#step-by-step-manual)
* [how to run the code with the prepared script](#use-prepared-scripts)


### Using pretrained models
For results to be easily reproducible, previously trained GRU models on different environments have been provided. You can simply use them to train new QBNs and reproduce the results presented in the paper. Models are accessible through this directory: ```results/Atari/```. The GRU cell size can be determined from the models' path, e.i. if a model is saved in a folder named as ```gru_32```, then the GRU cell size is 32. 
Having the pretrained GRU model, you can go to [how to run the code step by step](#step-by-step-manual) to start training the QBNs.

### A summary of results
Presenting the Mode Counter Environments(MCE) results, number of states and observations of the MMs extracted from the MMNs both before and after minimization. Moore Machine extraction for MCE(table 1 in paper):

|   Game	|   B<sub>h</sub>, B<sub>f</sub>    |   Fine-Tuning Score |  Before Minimization	|   After Minization    |
|:---------:|:---------------------------------:|:-------------------:|:-----------------------:|:---------------------:|
|          |                                  |  Before(%), After(%)|&#124;H&#124;, &#124;O&#124;, Accuracy(%)| &#124;H&#124;, &#124;O&#124;, Accuracy(%)|
|   Amnesia	|   4,4	|  0.98, 1 |    7, 5, 1 |  4, 4, 1 	|
|   Amnesia	|   4,8	|  0.99, 1 	|   7, 7, 1	|  4, 4, 1 	|
|  Amnesia 	|   8,4	|   1, -	|   6, 5, 1	|  4, 4, 1 	|
|   Amnesia	|   8,8	|   0.99, 1	|   7, 7, 1	|  4, 4, 1 	|
|   Blind	|   4,4	|  1, - 	|   12, 6, 1|  10, 1, 1	|
|   Blind	|   4,8	|  1, - 	|   12, 8, 1|  10, 1, 1	|
|  Blind 	|   8,4	|  1, - 	|   5, 6, 1|   10, 1, 1	|
|  Blind 	|   8,8	|  0.78, 1 	|   13, 8, 1|  10, 1, 1	|
|  Tracker 	|   4,4	|  0.98, 0.98 	|   58, 5, 0.98|  50, 4, 0.98|
|   Tracker	|   4,8	|  0.99, 1 	|   23, 5, 1|   10, 4, 1|
|   Tracker	|   8,4	|  0.98, 1 	|   91, 5, 1|   10, 4, 1|
|  Tracker 	|   8,8	|  0.99, 1 	|   85, 5, 1|   10, 4, 1|


The below table presents the test results for the trained RNNs giving the accuracy over a test set of 100 strings drawn from the same distribution as used for training. Moore Machine extraction for Tomita grammar(table 2 in paper):

|   Grammar	|   RNN Accuracy(%) | B<sub>h</sub>    |   Fine-Tuning Accuracy(%) |  Before Minimization	|   After Minization    |
|:---------:|:-----------------:|:---------------------------------:|:-------------------:|:-----------------------:|:---------------------:|
|          |                    |             |  Before, After|&#124;H&#124;, Accuracy(%)| &#124;H&#124;, Accuracy(%)|
|   1	|   100	|  8    |   100, -  |  13, 100 	|2, 100|
|   1	|   100	|  16   |   100, -  |  28, 100 	|2, 100|
|   2   |   100	|  8 	|   100, -	|  13, 100 	|3, 100|
|   2   |   100	|  16 	|   100, -	|  14, 100 	|3, 100|
|   3   |   100	|  8 	|   100, -	|  34, 100 	|5, 100|
|   3   |   100	|  16 	|   100, -	|  39, 100 	|5, 100|
|   4   |   100	|  8 	|   100, -	|  17, 100 	|4, 100|
|   4   |   100	|  16 	|   100, -	|  18, 100 	|4, 100|
|   5   |   100	|  8 	|   95, 96	|  192, 96 	|115, 96|
|   5   |   100	|  16 	|   100, -	|  316, 100 |4, 100|
|   6   |   99	|  8 	|   98, 98	|  100, 98 	|12, 98|
|   6   |   99	|  16 	|   99, 99	|  518, 99 	|11, 99|
|   7   |   100	|  8 	|   100, -	|  25, 100 	|5, 100|
|   7   |   100	|  16 	|   100, -	|  107, 100	|5, 100|


More experiments on control tasks have been done. Results are presented in the following table:

|Game(# of actions)|B<sub>h</sub>, B<sub>f</sub>    |  Before Minimization	|   After Minization    |
|:---:|:---:|:-------------------:|:---------------------:|
|          |               |&#124;H&#124;, &#124;O&#124;, Score| &#124;H&#124;, &#124;O&#124;, Score|
|CartPole(2)|   8, 8 	|10, 27, 500|5, 25, 500|
|LunarLander(4)|   128, 100 	|2550, 2197, 172|75, 77, 134|
|LunarLander(4)|   128, 400 	|2194, 1996, 201|27, 37, 205|


 This table shows the performance of the trained MMNs before and after finetuning for different combinations of B<sub>h</sub> and B<sub>f</sub>. A few more games investigated and the results are added to the table 3 of the paper:

|Game(# of actions)|RNN(score)|B<sub>h</sub>, B<sub>f</sub>    |   Fine-Tuning Score |  Before Minimization	|   After Minization    |
|:---:|:---:|:---:|:-------------------:|:-----------------------:|:---------------------:|
|          |              |                   |  Before, After|&#124;H&#124;, &#124;O&#124;, Score| &#124;H&#124;, &#124;O&#124;, Score|
|Pong(3)|21|  64, 100 	|20, 21|380, 374, 21|4, 12, 21|
|Pong(3)|21|  64, 400 	|20, 21|373, 372, 21|3, 10, 21|
|Pong(3)|21|  128, 100 |20, 21|383, 373, 21|3, 12, 21|
|Pong(3)|21|  128, 400 |20, 21|379, 371, 21|3, 11, 21|
|Freeway(3)|21| 64, 100 |21, -|1, 1, 21|1, 1, 21|
|Freeway(3)|21| 64, 400 |21, -|1, 1, 21|1, 1, 21|
|Freeway(3)|21| 128, 100|21, -|1, 1, 21|1, 1, 21|
|Freeway(3)|21| 128, 400|21, -|1, 1, 21|1, 1, 21|
|Breakout(4)|  773|  64, 100 |32, 423|1898, 1874, 423|8, 30, 423|
|Breakout(4)|  773|  64, 400 |25, 415|1888, 1871, 415|8, 30, 415|
|Breakout(4)|  773|  128, 100|41, 377|1583, 1514, 377|11, 27, 377
|Breakout(4)|  773|  128, 400|85, 379|1729, 1769, 379|8, 30, 379|
|Space Invaders(4)| 1820|   64, 100 |520, 1335|1495, 1502, 1335|8, 29, 1335|
|Space Invaders(4)| 1820|   64, 400 |365, 1235|1625, 1620, 1235|12, 29, 1235|
|Space Invaders(4)| 1820|   128, 100|390, 1040|1563, 1457, 1040|12, 35, 1040|
|Space Invaders(4)| 1820|   128, 400|520, 1430|1931, 1921, 1430|6, 27, 1430|
|Bowling(6)|      60|   64, 100 |60, -|49, 1, 60|33, 1, 60|
|Bowling(6)|      60|   64, 400 |60, -|49, 1, 60|33, 1, 60|
|Bowling(6)|      60|   128, 100|60, -|26, 1, 60|24, 1, 60|
|Bowling(6)|      60|   128, 400|60, -|26, 1, 60|24, 1, 60|
|Boxing(18)|   100|   64, 100 |94, 100|1173, 1167, 100|13, 79, 100|
|Boxing(18)|   100|   64, 400 |98, 100|2621, 2605, 100|14, 119, 100|
|Boxing(18)|   100|   128, 100|94, 97|2499, 2482, 97|14, 106, 97|
|Boxing(18)|   100|   128, 400|97, 100|1173, 1169, 100|14, 88, 100|
|Chopper Command(18) |   | 64, 100 | 4000 | 3710, 3731, 4000 | 38, 182, 1890| 



### Code walk-through
To run the code, there are several parameters that should be set. Below is a list of them:
>generate_bn_data: used to generate data for QBNs training and testing\
generate_max_steps: Maximum number of steps to be used for data generation\
gru_size: number of GRU cells\
bhx_size: size of the hidden state space QBN\
ox_size: size of the observation space QBN\
train_epochs: number of training episodes\
bhx_train: use it if you want to train the BHX net\
bhx_test: use it if you want to test the BHX net\
ox_train: use it if you want to train the OX net\
ox_test: use it if you want to test the OX net\
bgru_train: use it if you want to finetune the GRU net using the QBNs\
bgru_test: use it if you want to test the finetuned GRU net using the QBNs\
generate_fsm: use it if you want to generate the minimized FSM\
evaluate_fsm: use it if you want to evaluate the minimized FSM
>

By having the proper set of parameters, now you can run the ```main_atari.py``` code to start the process. For a step by step manual click [here](#step-by-step-manual).

### Step by step manual
In the first step, the given model should be tested to see if it gets relatively good results or not. This command is to be used to test how the model performs during the test:
<br/>```python main_atari.py --env **ENVIRONMENT** --gru_test --gru_size 32```

Before training the QBNs, the discrete data should be generated. This means that quantizing the continuous data into a discrete form. This can be done by the following command:
<br/>```python main_atari.py --env **ENVIRONMENT** --generate_bn_data --gru_size 32 --generate_max_steps 100```

Now, the hidden state QBN should be trained based on the previously trained RNN(**GRU**) model and using the data generated in the last step. To do this, the following command should be run to train the Bottleneck Hidden State(BHX) net:
<br/>```python main_atari.py --env **ENVIRONMENT** --bhx_train --bhx_size 64 --gru_size 32 --generate_max_steps 100```

After it's done, the model and plots will be saved here:
<br/>```results/Atari/**ENVIRONMENT**/gru_32_bhx_64/```

Now, it's time to test the new model that was trained using the QBNs(BHX net). Run the following command:
<br/>```python main_atari.py --env **ENVIRONMENT** --bhx_test --bhx_size 64 --gru_size 32 --generate_max_steps 100```

Now, the observation QBN should be trained. To do this, the following command should be run to train the Bottleneck Observation State(OX) net:
<br/>```python main_atari.py --env **ENVIRONMENT** --ox_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100```

After it's done, the model and plots will be saved here:
<br/>```results/Atari/**ENVIRONMENT**/gru_32_ox_100/```

When the OX net is trained, it can be tested. Run the following command to do so:
<br/>```python main_atari.py --env **ENVIRONMENT** --ox_test --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100```

Having the QBNs, it's time to fine-tune the RNN(**GRU**) model based on them. Following command would do that:
<br/>```python main_atari.py --env **ENVIRONMENT** --bgru_train --ox_size 100 --bhx_size 64 --gru_size 32 --generate_max_steps 100```

When the fine-tuning is done, model and plots will be saved here:
<br/>```results/Atari/**ENVIRONMENT**/gru_32_hx_(64,100)_bgru```

In this step the trained model in previous step is going to be tested by the following command:
<br/>```python main_atari.py --env **ENVIRONMENT** --bgru_test --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100```

Congrats, you've made it so far :). It is the final step. Here the final results will be converted into a finite state machine explanation text file. Run the following command for that:
<br/>```python main_atari.py --env **ENVIRONMENT** --generate_fsm --bhx_size 64 --ox_size 100 --gru_size 32 --generate_max_steps 100```

And the FSM explanation files will be saved as text files here:
<br/>```results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/```
<br/>In this directory two most important, files containing the observation space and hidden state space beبore minimization(in the file named as: "fsm.txt") and after minimization(in the file named as: "minimized_moore_machine.txt")


### Use prepared scripts

Instead of going through the step by step manual described above, one can use the prepared scripts. This script starts from testing a given GRU model on a defined environment and ends by generating the FSM. One can do this by simply running:
<br/>```sh run_atari.sh``` 
