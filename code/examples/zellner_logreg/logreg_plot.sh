#!/bin/bash

fldr_figs='figs'
fldr_res='results'
beta="0.01"
i0="1."
f_rate="15"
graddiag="False"
for dnm in 'adult' #'santa100K' 'webspam'
  do
    python3 plot.py $dnm $fldr_figs $fldr_res $beta $i0 $f_rate $graddiag
  done


'''
santa100K
--------------
beta="0.01"
i0="1."
f_rate="30"
graddiag="False"

adult
---------------
beta="0.01"
i0="1."
f_rate="15"
graddiag="False"
'''
