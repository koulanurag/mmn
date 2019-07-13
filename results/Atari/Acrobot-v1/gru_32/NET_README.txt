********Net Information********

MMNet(
  (obx_net): ObsQBNet(
    (encoder): Sequential(
      (0): Linear(in_features=6, out_features=128, bias=True)
      (1): Tanh()
      (2): Linear(in_features=128, out_features=16, bias=True)
      (3): TernaryTanh()
    )
    (decoder): Sequential(
      (0): Linear(in_features=16, out_features=128, bias=True)
      (1): Tanh()
      (2): Linear(in_features=128, out_features=6, bias=True)
      (3): ReLU6()
    )
  )
  (gru_net): GRUNet(
    (layer1): Linear(in_features=6, out_features=12, bias=True)
    (layer2): Linear(in_features=12, out_features=6, bias=True)
    (gru): GRUCell(6, 32)
    (critic_linear): Linear(in_features=32, out_features=1, bias=True)
    (actor_linear): Linear(in_features=32, out_features=3, bias=True)
  )
  (bhx_net): HxQBNet(
    (encoder): Sequential(
      (0): Linear(in_features=32, out_features=256, bias=True)
      (1): Tanh()
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): Tanh()
      (4): Linear(in_features=128, out_features=32, bias=True)
      (5): TernaryTanh()
    )
    (decoder): Sequential(
      (0): Linear(in_features=32, out_features=128, bias=True)
      (1): Tanh()
      (2): Linear(in_features=128, out_features=256, bias=True)
      (3): Tanh()
      (4): Linear(in_features=256, out_features=32, bias=True)
      (5): Tanh()
    )
  )
)

INFO:
time_taken : 9.0702
