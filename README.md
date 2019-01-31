# Introduction
In this document, a step by step manual on how to work with the *LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS*'s code is described. A simple environment such as TomitaB has been used as a toy example, but the main procedure would be the same with different environments, such as GoldRush, Pong etc.

## Step 0
In this step, the trajectory data will be created. The data model will be saved in the folder named after the environment in this directory: `results/Atari/**ENVIRONMENT**/trajectories_data.p`.
Use the below command to generate and save the data:<br />
`CUDA_VISIBLE_DEVICES=0 python main_gold_rush.py --env TomitaB-v0 --generate_train_data`

The output would be something similar to this:
>[root -> generate_trajectories]  Generating data .. <br />
>[root -> generate_trajectories]  Batch:0 Ep: 0 Reward:1<br />
>[root -> generate_trajectories]  Batch:0 Ep: 1 Reward:1<br />
>[root -> generate_trajectories]  Batch:0 Ep: 2 Reward:1<br />
>[root -> generate_trajectories]  Batch:0 Ep: 3 Reward:1<br />
>.<br />
>.<br />
>.<br />
>[root -> generate_trajectories]  Batch:99 Ep: 28 Reward:1<br />
>[root -> generate_trajectories]  Batch:99 Ep: 29 Reward:1<br />
>[root -> generate_trajectories]  Batch:99 Ep: 30 Reward:1<br />
>[root -> generate_trajectories]  Batch:99 Ep: 31 Reward:1<br />
>[root -> generate_trajectories]  Average Performance: 1.0


## Step 1
Train the RNN(**GRU**) network from **scratch**. This training doesn't include the QBNs. The aim of this step is to train a model to be used for training the QBNs.
Use the command below to do so:<br />
`CUDA_VISIBLE_DEVICES=0 python main_gold_rush.py --env TomitaB-v0 --gru_train --gru_size 10 --generate_max_steps 100`

The output would be something similar to this:
>[root -> generate_trajectories]  Loading Saved data .. <br/>
>[root -> <module>]  Training GRU!<br/>
>[gru_nn -> train]  Padding Sequences ...<br/>
>[gru_nn -> train]  epoch: 0 batch: 0 actor loss: 19.2468<br/>
>.<br/>
>.<br/>
>.<br/>
>[root -> generate_trajectories]  Batch:499 Ep: 27 Reward:1<br/>
>[root -> generate_trajectories]  Batch:499 Ep: 28 Reward:1<br/>
>[root -> generate_trajectories]  Batch:499 Ep: 29 Reward:1<br/>
>[root -> generate_trajectories]  Batch:499 Ep: 30 Reward:1<br/>
>[root -> generate_trajectories]  Batch:499 Ep: 31 Reward:1<br/>
>[root -> generate_trajectories]  Average Performance: 1.0

Also, there some plots to show how was the training. They will be save here: `results/Atari/**ENVIRONMENT**/**MODEL**/plots/`


## Step 2
This step is to be used to test how was training. Use the command below for that:
`CUDA_VISIBLE_DEVICES=0 python main_gold_rush.py --env TomitaB-v0 --gru_test --gru_size 10`

Here is the output:
>[root -> generate_trajectories]  Loading Saved data ..<br/> 
>[root -> <module>]  Testing GRU!<br/>
>[gru_nn -> test]  Episode =>0 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 40<br/>
>[gru_nn -> test]  Episode =>1 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 8<br/>
>.<br/>
>.<br/>
>.<br/>
>3/2019 05:38:14 PM [INFO ] [gru_nn -> test]  Episode =>17 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 26<br/>
>[gru_nn -> test]  Episode =>18 Score=> 1 Actions=> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 15<br/>
>[gru_nn -> test]  Episode =>19 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 48<br/>
>[root -> <module>]  Average Performance:1.0

This simply tells what is the model actions, ground truth(as episodes), score, and model's final accuracy.


## Step 3
Now QBNs data should be generated. This means that quantizing the continuous data(which are built in step 0) into a discrete form. This can be done by the following command:
`CUDA_VISIBLE_DEVICES=0 python main_gold_rush.py --env TomitaB-v0 --generate_bn_data --gru_size 10 --generate_max_steps 100`
And the output is like:
>[root -> <module>]  Generating Data-Set for Later Bottle Neck Training<br/>
>[root -> generate_bottleneck_data]  Data Sizes:<br/>
>[root -> generate_bottleneck_data]  Hx Train:589 Hx Test:269 Obs Train:589 Obs Test:2<br/>
[root -> generate_trajectories]  Loading Saved data ..

## Step 4
Now the QBNs can be trained based on the RNN(**GRU**) model trained earlier and the data generated in previous step. To do this, the following command should be run to train the BX net:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --bhx_train --bhx_size 8 --gru_size 10 --generate_max_steps 100`

Output is like:
>[gru_nn -> test]  Episode =>0 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 40<br/>
[gru_nn -> test]  Episode =>1 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 8<br/>
[gru_nn -> test]  Episode =>2 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 38<br/>
[gru_nn -> test]  Episode =>3 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 19<br/>
[gru_nn -> test]  Episode =>4 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 30<br/>
[root -> <module>]  Reward Threshold:1.0<br/>
[root -> <module>]  Loading Data-Set<br/>
[root -> generate_bottleneck_data]  Data Sizes:<br/>
[root -> generate_bottleneck_data]  Hx Train:589 Hx Test:269 Obs Train:589 Obs Test:2<br/>
[root -> <module>]  Training HX SandGlassNet!<br/>
venv/lib/python3.5/site-packages/torch/nn/functional.py:1320: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.<br/>
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")<br/>
[qbn -> train]  epoch: 0 batch: 0 loss: 0.786866<br/>
[qbn -> train]  epoch: 0 batch: 1 loss: 0.783952<br/>
.<br/>
.<br/>
.<br/>
[qbn -> train]  epoch: 204 batch: 14 loss: 0.004681<br/>
[qbn -> train]  epoch: 204 batch: 15 loss: 0.004578<br/>
[qbn -> train]  epoch: 204 batch: 16 loss: 0.004540<br/>
[qbn -> train]  epoch: 204 batch: 17 loss: 0.004639<br/>
[qbn -> train]  epoch: 204 batch: 18 loss: 0.004656<br/>
[qbn -> train]  Bottle Net Model Saved!<br/>
01/23/2019 06:02:56 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:56 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
01/23/2019 06:02:57 PM [DEBUG] [matplotlib.axes._base -> _update_title_position]  update_title_pos<br/>
[tools -> plot_data]  Plot Saved! - results/Atari/TomitaB-v0/gru_10_bhx_8/Plots<br/>
[qbn -> train]  epoch: 204 test loss: 0.004970 best perf i: 0 min loss i: 186

After it's done, the model will be saved here: `results/Atari/**ENVIRONMENT**/**MODEL**/model.p`.


## Step 5
Now it's time to test the new model that was trained using the QBNs(BX net). Run the following command:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --bhx_test --bhx_size 8 --gru_size 10 --generate_max_steps 100`
The output:
>[gru_nn -> test]  Episode =>0 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 40<br/>
[gru_nn -> test]  Episode =>1 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 8<br/>
[gru_nn -> test]  Episode =>2 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 38<br/>
[gru_nn -> test]  Episode =>3 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 19<br/>
[gru_nn -> test]  Episode =>4 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 30<br/>
[root -> <module>]  Reward Threshold:1.0<br/>
[root -> <module>]  Loading Data-Set<br/>
[root -> generate_bottleneck_data]  Data Sizes:<br/>
[root -> generate_bottleneck_data]  Hx Train:589 Hx Test:269 Obs Train:589 Obs Test:2<br/>
[root -> <module>]  Testing  HX SandGlassNet<br/>
venv/lib/python3.5/site-packages/torch/nn/functional.py:1320: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.<br/>
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")<br/>
[root -> <module>]  MSE :0.004303866997361183


## Step 6
Now the QBNs can be trained based on the RNN(**GRU**) model trained earlier and the data generated in previous steps. To do this, the following command should be run to train the OX net:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --ox_train --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`


## Step 7
Now it's time to test the new model that was trained using the QBNs(OX net). Run the following command:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --ox_test --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`


## Step 8
Having the QBNs, it's time to fine-tune the RNN(**GRU**) model. Running the following command would do that:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --bgru_train --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

The output:
>[gru_nn -> test]  Episode =>0 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 40<br/>
[gru_nn -> test]  Episode =>1 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 8<br/>
[gru_nn -> test]  Episode =>2 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 38<br/>
[gru_nn -> test]  Episode =>3 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 19<br/>
[gru_nn -> test]  Episode =>4 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 30<br/>
[gru_nn -> test]  Episode =>5 Score=> 1 Actions=> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 32<br/>
[gru_nn -> test]  Episode =>6 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 16<br/>
[gru_nn -> test]  Episode =>7 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 33<br/>
[gru_nn -> test]  Episode =>8 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 40<br/>
[gru_nn -> test]  Episode =>9 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 46<br/>
[root -> <module>]  Training Binary GRUNet!<br/>
[root -> generate_trajectories]  Loading Saved data ..<br/> 
[bgru_nn -> train]  Padding Sequences ...<br/>
venv/lib/python3.5/site-packages/torch/nn/functional.py:1320: UserWarning: nn.functional.tanh is deprecated. Use torch.tanh instead.<br/>
  warnings.warn("nn.functional.tanh is deprecated. Use torch.tanh instead.")<br/>
[bgru_nn -> train]  epoch 0 Test Performance: 1.000000<br/>
[bgru_nn -> train]  Binary GRU Model Saved!<br/>
[bgru_nn -> train]  Optimal Performance achieved!!!<br/>
[bgru_nn -> train]  Exiting!

NOTE: Since the model already have the accuracy of 1.00, it doesn't need to be trained again, so the training stops right away. **In simple environments, this happens more often.**

After training is done, the model will be saved here: `results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru`


## Step 9
Here the trained model in previous step is going to be tested by the following command:
`CUDA_VISIBLE_DEVICES=0  python main_gold_rush.py --env TomitaB-v0 --bgru_test --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`
The output:
>[bgru_nn -> test]  Episode =>0 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 40<br/>
[bgru_nn -> test]  Episode =>1 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 8<br/>
[bgru_nn -> test]  Episode =>2 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 38<br/>
[bgru_nn -> test]  Episode =>3 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 19<br/>
.<br/>
.<br/>
.<br/>
[bgru_nn -> test]  Episode =>46 Score=> 1 Actions=> [0] ActionCount=> 1<br/>
[bgru_nn -> test]  Episode =>47 Score=> 1 Actions=> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ActionCount=> 38<br/>
[bgru_nn -> test]  Episode =>48 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 0, 0, 0] ActionCount=> 10<br/>
[bgru_nn -> test]  Episode =>49 Score=> 1 Actions=> [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] ActionCount=> 50<br/>
[root -> <module>]  Average Performance: 1.0

As it's been printed, the accuracy is 1.00. In each test case, the ground truth(as episodes), scores, actions taken, and results are printed.


## Step 10
Congrats, you've made it so far. It's the final step. Here the final results will be converted into a finite state machine. Run the following command for that:
`python main_gold_rush.py --env TomitaB-v0 --generate_fsm --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

The output:
>.<br/>
.<br/>
.<br/>
[moore_machine -> extract_from_nn]  Episode:8 Reward: 1<br/> 
[moore_machine -> extract_from_nn]  Episode:9 Reward: 1<br/> 
[moore_machine -> extract_from_nn]  Average Reward:1.0<br/>
[moore_machine -> evaluate]  Episode => 0 Score=> 1<br/>
[moore_machine -> evaluate]  None state encountered!<br/>
[moore_machine -> evaluate]  Exiting the script!

And the FSM models are saved as text files here: `results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/fsm.txt`.