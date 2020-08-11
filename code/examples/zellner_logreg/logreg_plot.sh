#!/bin/bash

fldr_figs='figs'
fldr_res='results'
for beta in "0.9" #"0.7"
do
for i0 in "0.1"
do
for f_rate in "0.1" #"0" "30"
do
for graddiag in  "False"
do
  for structured in  "False"
  do
for dnm in 'phish'
  do
    #echo $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
    python3 plot.py $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag $structured
  done
  done
done
done
done
done
