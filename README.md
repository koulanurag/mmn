# Introduction
In this document, a manual on how to work with the *LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS*'s code is described.

#### Step by step manual
---
In the first step, the trajectory data will be created. The data model will be saved in the folder named after the environment in this directory: <br/>`results/Atari/**ENVIRONMENT**/trajectories_data.p`

Use the below command to generate and save the data:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_train_data`

Now, train the RNN(**GRU**) network from **scratch**. This training does not include the QBNs. This step aims to train a model to be used for training the QBNs.
Use the command below to do so:
<br/>`python main_atari.py --env **ENVIRONMENT** --gru_train --gru_size 10 --generate_max_steps 100`

Also, some plots to showing how was the training will be saved here:<br />`results/Atari/**ENVIRONMENT**/**MODEL**/plots/`

This command is to be used to test how was training:
<br/>`python main_atari.py --env **ENVIRONMENT** --gru_test --gru_size 10`

Befor training the QBNs, the discrete data should be generated. This means that quantizing the continuous data(which are built in step 0) into a discrete form. This can be done by the following command:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_bn_data --gru_size 10 --generate_max_steps 100`

Now, the hidden state QBN can be trained based on the RNN(**GRU**) model trained earlier and the data generated in the previous step. To do this, the following command should be run to train the Bottleneck Hidden State(BHX) net:
<br/>`python main_atari.py --env **ENVIRONMENT** --bhx_train --bhx_size 8 --gru_size 10 --generate_max_steps 100`

After it's done, the model will be saved here:<br/>`results/Atari/**ENVIRONMENT**/**MODEL**/model.p`

When BHX net training is done, it's time to test the new model that was trained using the QBNs(BHX net). Run the following command:<br/>
`python main_atari.py --env **ENVIRONMENT** --bhx_test --bhx_size 8 --gru_size 10 --generate_max_steps 100`

Now, the observation QBN can be trained based on the RNN(**GRU**) model trained earlier and the data generated in previous steps. To do this, the following command should be run to train the Bottleneck Observation State(OX) net:<br/>
`python main_atari.py --env **ENVIRONMENT** --ox_train --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`

When the OX net is trained, it can be tested. Run the following command to do so:<br/>
`python main_atari.py --env **ENVIRONMENT** --ox_test --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`

Having the QBNs, it's time to fine-tune the RNN(**GRU**) model based on the QBNs. Running the following command would do that:<br/>
`python main_atari.py --env **ENVIRONMENT** --bgru_train --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

When the fine-tuning is done, model will be saved here:<br/> `results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru`

In this step the trained model in previous step is going to be tested by the following command:<br/>
`python main_atari.py --env **ENVIRONMENT** --bgru_test --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

Congrats, you have made it so far :). It is the final step. Here the final results will be converted into a finite state machine explanation text file. Run the following command for that:<br/>
`python main_atari.py --env **ENVIRONMENT** --generate_fsm --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

And the FSM explanation files will be saved as text files here:<br/>`results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/fsm.txt`

---
#### Use prepared scripts
---
Instead of going through the step by step manual described above, one can use the prepared scripts. This script starts from training a GRU model on a defined environment and ends by generating the FSM. One can do this by running the `run_atari.sh`. 
