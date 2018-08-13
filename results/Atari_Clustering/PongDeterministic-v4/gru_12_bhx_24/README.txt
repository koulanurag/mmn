******Net Information********

HxBNNet(
  (bin_encoder): Sequential(
    (0): Linear(in_features=12, out_features=120, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=120, out_features=72, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=72, out_features=24, bias=True)
    (5): TernaryTanh()
  )
  (bin_decoder): Sequential(
    (0): Linear(in_features=24, out_features=72, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=72, out_features=120, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=120, out_features=12, bias=True)
    (5): Tanh()
  )
)

INFO:
time_taken : 100.8717
