******Net Information********

BNNet(
  (bin_encoder): Sequential(
    (0): Linear(in_features=1, out_features=12, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=12, out_features=4, bias=True)
    (3): BinaryTanh(
      (hardtanh): Tanh()
    )
  )
  (bin_decoder): Sequential(
    (0): Linear(in_features=4, out_features=12, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=12, out_features=1, bias=True)
    (3): Tanh()
  )
)

INFO:
time_taken : 242.1698
