******Net Information********

BNNet(
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

INFO:
time_taken : 589.5795
