#!/bin/bash

fldr_figs='figs'
fldr_res='results'
for beta in "0.9"
do
for i0 in "0.1" "1.0" "10.0"
do
for f_rate in "15" "0" 
do
for graddiag in "True"
do
  for structured in "False"
  do
for dnm in 'webspam'
  do
    #echo $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
    python3 plot.py $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag $structured
  done
  done
done
done
done
done
