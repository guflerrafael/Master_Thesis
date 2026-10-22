```
            tier                            question  seeds  min/epoch      AP@5 best     AP@5 last5     AP@10 best    AP@10 last5  Δ AP@5 best vs ref  Δ AP@10 best vs ref  Δ AP@5 last5 vs ref  Δ AP@10 last5 vs ref
experiment                                                                                                                                                                                                            
ref            1             reference = v1 (hourly)      3       1.19  0.346 ± 0.060  0.309 ± 0.039  0.667 ± 0.016  0.665 ± 0.016               0.000                0.000                0.000                 0.000
sigma05        3     narrower targets (sigma 0.5 px)      3       1.19  0.390 ± 0.072  0.342 ± 0.059  0.674 ± 0.031  0.669 ± 0.026               0.044                0.007                0.033                 0.004
sigma2         3          wider targets (sigma 2 px)      3       1.19  0.327 ± 0.030  0.291 ± 0.016  0.637 ± 0.008  0.637 ± 0.007              -0.019               -0.031               -0.018                -0.028
loss_bce       3  weighted BCE instead of focal loss      3       1.20  0.415 ± 0.052  0.343 ± 0.023  0.668 ± 0.007  0.667 ± 0.007               0.069                0.001                0.035                 0.002
loss_mse       3           MSE instead of focal loss      3       1.23  0.428 ± 0.014  0.347 ± 0.012  0.655 ± 0.011  0.660 ± 0.008               0.082               -0.012                0.038                -0.005
neg0           3                   no hard negatives      3       0.60  0.358 ± 0.043  0.273 ± 0.044  0.644 ± 0.025  0.650 ± 0.012               0.012               -0.023               -0.035                -0.015
neg05          3     0.5 hard negatives per positive      3       0.91  0.337 ± 0.039  0.283 ± 0.028  0.666 ± 0.022  0.670 ± 0.013              -0.009               -0.002               -0.025                 0.005
neg2           3       2 hard negatives per positive      3       1.78  0.406 ± 0.024  0.301 ± 0.036  0.683 ± 0.028  0.686 ± 0.011               0.060                0.016               -0.008                 0.021
ignore0        3       no ignore regions in training      3       1.21  0.385 ± 0.071  0.324 ± 0.052  0.682 ± 0.015  0.675 ± 0.010               0.039                0.015                0.016                 0.010
ignore4        3      ignore radius 4 px in training      3       1.20  0.328 ± 0.021  0.286 ± 0.025  0.653 ± 0.029  0.649 ± 0.013              -0.018               -0.015               -0.023                -0.016
target100      3              targets >= 100 ha only      3       0.76  0.381 ± 0.030  0.299 ± 0.030  0.654 ± 0.009  0.662 ± 0.020               0.035               -0.013               -0.009                -0.003
noflip         3                no flip augmentation      3       1.21  0.615 ± 0.012  0.580 ± 0.045  0.690 ± 0.018  0.657 ± 0.039               0.269                0.023                0.271                -0.008
```
