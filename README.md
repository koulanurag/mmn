# Introduction
In this document, a step by step manual on how to work with the *LEARNING FINITE STATE REPRESENTATIONS OF RECURRENT POLICY NETWORKS*'s code is described. The procedure would be the same in different environments.

### Step 0 - Generate training data
In this step, the trajectory data will be created. The data model will be saved in the folder named after the environment in this directory: <br/>`results/Atari/**ENVIRONMENT**/trajectories_data.p`

Use the below command to generate and save the data:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_train_data`

### Step 1 - Train the GRU
Train the RNN(**GRU**) network from **scratch**. This training does not include the QBNs. This step aims to train a model to be used for training the QBNs.
Use the command below to do so:
<br/>`python main_atari.py --env **ENVIRONMENT** --gru_train --gru_size 10 --generate_max_steps 100`

Also, some plots to show how was the training will be saved here:<br />`results/Atari/**ENVIRONMENT**/**MODEL**/plots/`

### Step 2 - Test the GRU
This step is to be used to test how was training. Use the command below for that:
<br/>`python main_atari.py --env **ENVIRONMENT** --gru_test --gru_size 10`

### Step 3 - Generate QBN data
Now QBNs data should be generated. This means that quantizing the continuous data(which are built in step 0) into a discrete form. This can be done by the following command:
<br/>`python main_atari.py --env **ENVIRONMENT** --generate_bn_data --gru_size 10 --generate_max_steps 100`

### Step 4 - BHX net training
Now, the hidden state QBN can be trained based on the RNN(**GRU**) model trained earlier and the data generated in the previous step. To do this, the following command should be run to train the Bottleneck Hidden State(BHX) net:
<br/>`python main_atari.py --env **ENVIRONMENT** --bhx_train --bhx_size 8 --gru_size 10 --generate_max_steps 100`

After it's done, the model will be saved here:<br/>`results/Atari/**ENVIRONMENT**/**MODEL**/model.p`

### Step 5 - BHX net test
Now it's time to test the new model that was trained using the QBNs(BX net). Run the following command:<br/>
`python main_atari.py --env **ENVIRONMENT** --bhx_test --bhx_size 8 --gru_size 10 --generate_max_steps 100`

### Step 6 - OX net training
Now the observation QBN can be trained based on the RNN(**GRU**) model trained earlier and the data generated in previous steps. To do this, the following command should be run to train the Bottleneck Observation State(ox) net:<br/>
`python main_atari.py --env **ENVIRONMENT** --ox_train --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`

### Step 7 - OX net test
It's time to test the new model that was trained using the QBNs(OX net). Run the following command:<br/>
`python main_atari.py --env **ENVIRONMENT** --ox_test --ox_size 8 --bhx_size 8 --gru_size 10 --generate_max_steps 100`

### Step 8 - Fine-tuning the GRU
Having the QBNs, it's time to fine-tune the RNN(**GRU**) model. Running the following command would do that:<br/>
`python main_atari.py --env **ENVIRONMENT** --bgru_train --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

After training is done, the model will be saved here:<br/> `results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru`

### Step 9 - Test the fine-tuned GRU model
Here the trained model in previous step is going to be tested by the following command:<br/>
`python main_atari.py --env **ENVIRONMENT** --bgru_test --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

### Step 10 - Generating the Finite State Machine(FSM)
Congrats, you have made it so far :). It is the final step. Here the final results will be converted into a finite state machine explanation text file. Run the following command for that:<br/>
`python main_atari.py --env **ENVIRONMENT** --generate_fsm --bhx_size 8 --ox_size 8 --gru_size 10 --generate_max_steps 100`

And the FSM explanation files will be saved as text files here:<br/>`results/Atari/**ENVIRONMENT**/gru_10_hx_(8,1)_bgru/fsm.txt`
