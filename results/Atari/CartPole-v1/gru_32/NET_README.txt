********Net Information********

MMNet(
  (obx_net): ObsQBNet(
    (encoder): Sequential(
      (0): Linear(in_features=4, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=8, bias=True)
      (3): TernaryTanh()
    )
    (decoder): Sequential(
      (0): Linear(in_features=8, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=4, bias=True)
      (3): ReLU6()
    )
  )
  (gru_net): GRUNet(
    (layer1): Linear(in_features=4, out_features=4, bias=True)
    (layer2): Linear(in_features=4, out_features=4, bias=True)
    (gru): GRUCell(4, 32)
    (critic_linear): Linear(in_features=32, out_features=1, bias=True)
    (actor_linear): Linear(in_features=32, out_features=2, bias=True)
  )
  (bhx_net): HxQBNet(
    (encoder): Sequential(
      (0): Linear(in_features=32, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=32, bias=True)
      (3): Tanh()
      (4): Linear(in_features=32, out_features=8, bias=True)
      (5): TernaryTanh()
    )
    (decoder): Sequential(
      (0): Linear(in_features=8, out_features=32, bias=True)
      (1): Tanh()
      (2): Linear(in_features=32, out_features=64, bias=True)
      (3): Tanh()
      (4): Linear(in_features=64, out_features=32, bias=True)
      (5): Tanh()
    )
  )
)

INFO:
time_taken : 75.7775
