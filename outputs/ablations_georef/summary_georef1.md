```
             tier                                         question  seeds  min/epoch      AP@5 best     AP@5 last5     AP@10 best    AP@10 last5  Δ AP@5 best vs ref  Δ AP@10 best vs ref  Δ AP@5 last5 vs ref  Δ AP@10 last5 vs ref
experiment                                                                                                                                                                                                                          
ref             1                          reference = v1 (hourly)      3       1.25  0.604 ± 0.016  0.597 ± 0.015  0.687 ± 0.003  0.681 ± 0.007               0.000                0.000                0.000                 0.000
births_plus     1  raw input + clustering (nucleus births) channel      3       1.23  0.590 ± 0.047  0.577 ± 0.057  0.669 ± 0.044  0.661 ± 0.050              -0.014               -0.018               -0.020                -0.020
sigma05         3                  narrower targets (sigma 0.5 px)      3       1.24  0.604 ± 0.002  0.596 ± 0.004  0.681 ± 0.011  0.678 ± 0.009              -0.000               -0.006               -0.001                -0.003
neg2            3                    2 hard negatives per positive      3       1.86  0.618 ± 0.009  0.614 ± 0.009  0.705 ± 0.011  0.699 ± 0.009               0.014                0.018                0.017                 0.017
noflip          3                             no flip augmentation      3       1.25  0.633 ± 0.011  0.613 ± 0.021  0.686 ± 0.007  0.672 ± 0.015               0.028               -0.001                0.016                -0.009
```
