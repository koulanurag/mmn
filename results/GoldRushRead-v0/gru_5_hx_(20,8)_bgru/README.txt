******Net Information********

BinaryGRUNet(
  (obx_net): BNNet(
    (bin_encoder): Sequential(
      (0): Linear(in_features=4, out_features=5, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=5, out_features=8, bias=True)
      (3): BinaryTanh(
        (hardtanh): Hardtanh(min_val=-1, max_val=1)
      )
    )
    (bin_decoder): Sequential(
      (0): Linear(in_features=8, out_features=5, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=5, out_features=4, bias=True)
    )
  )
  (gru_net): GRUNet(
    (input_ff): Sequential(
      (0): Linear(in_features=1, out_features=4, bias=True)
      (1): ReLU()
    )
    (gru): GRUCell(4, 5)
    (actor_linear): Linear(in_features=5, out_features=4, bias=True)
  )
  (bhx_net): BNNet(
    (bin_encoder): Sequential(
      (0): Linear(in_features=5, out_features=5, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=5, out_features=20, bias=True)
      (3): BinaryTanh(
        (hardtanh): Hardtanh(min_val=-1, max_val=1)
      )
    )
    (bin_decoder): Sequential(
      (0): Linear(in_features=20, out_features=5, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=5, out_features=5, bias=True)
    )
  )
)

INFO:
time_taken : 71.1037
