#!/bin/bash

for f_rate in 0 30
  do
    for i0 in 0.1
    do
      for beta in 0.2
      do
        for dnm in "year"
        do
          python3 plot.py $dnm $i0 $f_rate $beta
        done
     done
  done
done
