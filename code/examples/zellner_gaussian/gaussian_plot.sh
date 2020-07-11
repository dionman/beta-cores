#!/bin/bash

n_trials=2
plot_every=5
fldr_plts='figs'
fldr_res='results'
for f_rate in 0 15 30
do
  python3 plot_kl.py $n_trials $plot_every $fldr_plts $fldr_res $f_rate
done
